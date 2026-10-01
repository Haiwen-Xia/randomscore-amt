"""Per-dataset teacher-forced loss and per-file instrument-agnostic Note F1."""

from contextlib import nullcontext
import logging

import mir_eval
import numpy as np
import torch
import torch.distributed as dist

from data.datasets import events_for_clip, load_annotation, read_indexes, read_mix
from tokenization.event2note import merge_zipped_note_events_and_ties_to_notes
from tokenization.note2event import mix_notes
from tokenization.vocabulary import PROGRAM_TO_CHANNEL
from .loss import make_batch, loss_parts


def autocast(device, precision):
    return (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if precision == "bf16" and device.type == "cuda"
        else nullcontext()
    )


def note_f1(reference, prediction):
    """YourMT3 onset_f: non-drum notes, 50 ms onset / 50 cent pitch, no offsets."""

    def valid(notes):
        return [
            n
            for n in notes
            if not n.is_drum
            and np.isfinite([n.onset, n.offset, n.pitch]).all()
            and n.onset >= 0
            and n.offset > n.onset
        ]

    reference, prediction = valid(reference), valid(prediction)
    if not reference:
        return float(
            "nan"
        )  # Like YourMT3, exclude files without pitched reference notes.
    if not prediction:
        return 0.0
    intervals = lambda notes: np.asarray(
        [[n.onset, max(n.offset, n.onset + 0.01)] for n in notes]
    )
    pitches = lambda notes: (
        440.0 * 2 ** ((np.asarray([n.pitch for n in notes]) - 69) / 12.0)
    )
    return float(
        mir_eval.transcription.precision_recall_f1_overlap(
            intervals(reference),
            pitches(reference),
            intervals(prediction),
            pitches(prediction),
            onset_tolerance=0.05,
            pitch_tolerance=50,
            offset_ratio=None,
        )[2]
    )


def segment_samples(audio, notes, starts, frames, sample_rate, spec):
    # Evaluation follows the task vocabulary; the caller keeps the complete
    # reference separately for instrument-agnostic F1, including out-of-vocab programs.
    notes = [
        n for n in notes if (128 if n.is_drum else n.program) in PROGRAM_TO_CHANNEL
    ]
    samples = []
    for start in starts:
        piece = audio[start : start + frames]
        piece = np.pad(piece, (0, frames - len(piece)))
        events, ties = events_for_clip(
            notes,
            start / sample_rate,
            frames / sample_rate,
            spec.get("allowed_programs"),
        )
        samples.append(
            {
                "audio": piece[None],
                "note_events": events,
                "tie_note_events": ties,
                "has_unannotated": False,
            }
        )
    return samples


@torch.no_grad()
def evaluate(model, tokenizer, config, device, train_entries=None, online=None):
    """Shard files without DistributedSampler padding; aggregate sums and counts."""
    was_training = model.training
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    evaluation = config["evaluation"]
    frames, sr = config["audio"]["input_frames"], config["audio"]["sample_rate"]
    batch_size = evaluation["batch_size"]
    scopes = []
    if train_entries is not None and evaluation["train_clips_per_dataset"]:
        for spec in config["data"]["train"]:
            scopes.append(("train", spec, train_entries[spec["name"]]))
    test_entries = read_indexes(config["data_root"], config["data"]["test"])
    scopes.extend(
        ("test", spec, test_entries[spec["name"]]) for spec in config["data"]["test"]
    )
    metrics = {}
    for scope, spec, entries in scopes:
        limit = (
            evaluation["train_clips_per_dataset"]
            if scope == "train"
            else evaluation["max_files"]
            if evaluation["max_files"] is not None
            else spec.get("max_files")
        )
        rng = np.random.default_rng(config["run"]["seed"])
        selected = np.arange(len(entries))
        if limit is not None:
            selected = rng.permutation(selected)[:limit]
        sums = np.zeros(4, dtype=np.float64)  # loss*segments, segments, file F1, files
        for file_index in selected[rank::world]:
            entry = entries[int(file_index)]
            notes = load_annotation(entry["notes_file"])["notes"]
            unsupported = sorted(
                {
                    n.program
                    for n in notes
                    if not n.is_drum and n.program not in PROGRAM_TO_CHANNEL
                }
            )
            if unsupported:
                logging.warning(
                    "%s: programs %s are outside the decoder vocabulary; excluded from loss targets, retained for Note F1",
                    spec["name"],
                    unsupported,
                )
            audio = read_mix(entry, sr)
            if scope == "train":
                # A fixed clip per file, independent of rank/world size.
                clip_rng = np.random.default_rng(
                    [config["run"]["seed"], int(file_index)]
                )
                starts = [int(clip_rng.integers(max(1, len(audio) - frames + 1)))]
            else:
                starts = list(range(0, max(1, len(audio)), frames))
                if evaluation["max_segments_per_file"] is not None:
                    starts = starts[: evaluation["max_segments_per_file"]]
            channel_segments = [[] for _ in range(model.encoder.output_channels)]
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                samples = segment_samples(audio, notes, batch_starts, frames, sr, spec)
                batch = make_batch(samples, tokenizer, config, device)
                with autocast(device, config["training"]["precision"]):
                    output = model(batch["audio"], batch["tokens"])
                    # Keep file/segment weighting stable when the final batch is short.
                    for i in range(len(samples)):
                        parts = loss_parts(
                            {
                                k: None if v is None else v[i : i + 1]
                                for k, v in output.items()
                            },
                            {k: v[i : i + 1] for k, v in batch.items()},
                        )
                        loss = sum(
                            total
                            / count.clamp_min(1)
                            * (
                                config["training"]["frame_loss_weight"]
                                if key == "frame"
                                else 1
                            )
                            for key, (total, count) in parts.items()
                        )
                        sums[0] += float(loss)
                        sums[1] += 1
                    if scope == "test":
                        generated = (
                            model.generate(batch["audio"], tokenizer.max_length)
                            .cpu()
                            .tolist()
                        )
                        for row, start in zip(generated, batch_starts):
                            for channel, ids in enumerate(row):
                                events, ties, active, _ = tokenizer.tokenizer.decode(
                                    ids, start_time=start / sr
                                )
                                channel_segments[channel].append(
                                    (events, ties, active, start / sr)
                                )
            if scope == "test":
                prediction = mix_notes(
                    [
                        merge_zipped_note_events_and_ties_to_notes(parts)[0]
                        for parts in channel_segments
                    ]
                )
                end = min(len(audio), starts[-1] + frames) / sr
                reference = [n for n in notes if n.onset < end]
                score = note_f1(reference, prediction)
                if np.isfinite(score):
                    sums[2] += score
                    sums[3] += 1
        reduced = torch.tensor(sums, device=device)
        if dist.is_initialized():
            dist.all_reduce(reduced)
        loss_sum, segments, f1_sum, files = reduced.cpu().tolist()
        if not segments:
            raise ValueError(f"evaluation selected no segments for {spec['name']}")
        prefix = f"eval/{scope}/{spec['name']}"
        metrics[f"{prefix}/loss"] = loss_sum / segments
        if scope == "test":
            metrics[f"{prefix}/note_f1"] = f1_sum / files if files else None
    if online is not None:
        count = evaluation["train_clips_per_dataset"]
        for name in online.names:
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            for index in range(rank, count, world):
                seed = int(
                    np.random.SeedSequence(
                        [config["run"]["seed"], 900001, index]
                    ).generate_state(1, dtype=np.uint64)[0]
                )
                samples = online.sample_from_seed(seed, route_name=name)[:1]
                batch = make_batch(samples, tokenizer, config, device)
                with autocast(device, config["training"]["precision"]):
                    parts = loss_parts(model(batch["audio"], batch["tokens"]), batch)
                    loss = sum(
                        total
                        / n.clamp_min(1)
                        * (
                            config["training"]["frame_loss_weight"]
                            if key == "frame"
                            else 1
                        )
                        for key, (total, n) in parts.items()
                    )
                totals[0] += loss
                totals[1] += 1
            if dist.is_initialized():
                dist.all_reduce(totals)
            if totals[1] > 0:
                metrics[f"eval/train/{name}/loss"] = float(totals[0] / totals[1])
    model.train(was_training)
    return metrics
