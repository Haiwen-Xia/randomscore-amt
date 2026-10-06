"""Small, resumable note-F1 benchmark for native RandomScore-AMT checkpoints."""

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.datasets import load_annotation, read_indexes, read_mix
from models.model import CompactModel
from tokenization.event2note import merge_zipped_note_events_and_ties_to_notes
from tokenization.note2event import mix_notes
from tokenization.note_event_dataclasses import Note
from tokenization.targets import TargetTokenizer
from training.evaluation import autocast, note_f1


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def selected_rows(root, spec, limit):
    entries = read_indexes(root, [spec])[spec["name"]]
    if limit is None:
        return entries
    subset_field = spec.get("subset_field")
    if subset_field is None:
        return entries[:limit]
    counts, selected = {}, []
    for entry in entries:
        subset = str(entry[subset_field])
        if counts.get(subset, 0) < limit:
            selected.append(entry)
            counts[subset] = counts.get(subset, 0) + 1
    return selected


def predict_notes(model, tokenizer, audio, config, device, batch_size, max_segments):
    frames = config["audio"]["input_frames"]
    sr = config["audio"]["sample_rate"]
    starts = list(range(0, max(1, len(audio)), frames))
    if max_segments is not None:
        starts = starts[:max_segments]
    channel_segments = [[] for _ in range(model.encoder.output_channels)]
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        pieces = [
            np.pad(
                audio[start : start + frames], (0, max(0, start + frames - len(audio)))
            )
            for start in batch_starts
        ]
        batch = torch.from_numpy(np.stack(pieces)[:, None]).to(device)
        with torch.no_grad(), autocast(device, config["training"]["precision"]):
            generated = model.generate(batch, tokenizer.max_length).cpu().tolist()
        for row, start in zip(generated, batch_starts):
            for channel, ids in enumerate(row):
                events, ties, active, _ = tokenizer.tokenizer.decode(
                    ids, start_time=start / sr
                )
                channel_segments[channel].append((events, ties, active, start / sr))
    notes = mix_notes(
        [
            merge_zipped_note_events_and_ties_to_notes(parts)[0]
            for parts in channel_segments
        ]
    )
    return notes, min(len(audio), starts[-1] + frames) / sr


def run(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.max_segments is not None and args.max_segments < 1:
        raise ValueError("--max-segments must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.resolve()
    config_path = args.config or checkpoint.parent / "config.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    suite = OmegaConf.to_container(OmegaConf.load(args.datasets), resolve=True)
    root = args.data_root or Path(config["data_root"])
    device = torch.device(args.device)
    output = args.output_dir or Path("outputs/benchmarks") / checkpoint.parent.name
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "checkpoint": file_identity(checkpoint),
        "config": file_identity(config_path),
        "datasets": file_identity(args.datasets),
        "data_root": str(root.resolve()),
        "limit_per_subset": args.limit,
        "max_segments_per_track": args.max_segments,
    }
    tokenizer = model = None

    def get_model():
        nonlocal tokenizer, model
        if model is None:
            print(f"Loading checkpoint {checkpoint} on {device}", flush=True)
            tokenizer = TargetTokenizer(**config["tokenization"])
            model = CompactModel(
                config["model"],
                tokenizer.num_tokens,
                config["training"]["frame_loss_weight"],
            ).to(device)
            state = torch.load(
                checkpoint, map_location="cpu", weights_only=False, mmap=True
            )
            if state.get("format_version") != 1:
                raise ValueError("expected a randomscore-amt checkpoint")
            model.load_state_dict(state["model"], strict=True)
            del state
            model.eval()
            print("Checkpoint loaded", flush=True)
        return model, tokenizer

    results = {}
    for spec in suite["datasets"]:
        name = spec["name"]
        rows = selected_rows(root, spec, args.limit)
        print(f"Evaluating {name}: {len(rows)} tracks", flush=True)
        records = []
        for position, row in enumerate(rows):
            source = {
                "audio": file_identity(row["mix_audio_file"]),
                "notes": file_identity(row["notes_file"]),
            }
            cache_path = output / "notes" / name / f"{position:05d}.json"
            cached = None
            if args.cache_notes and cache_path.is_file():
                candidate = json.loads(cache_path.read_text())
                if (
                    candidate.get("identity") == identity
                    and candidate.get("source") == source
                ):
                    cached = candidate
            if cached is None:
                audio = read_mix(row, config["audio"]["sample_rate"])
                loaded_model, loaded_tokenizer = get_model()
                predicted, end = predict_notes(
                    loaded_model,
                    loaded_tokenizer,
                    audio,
                    config,
                    device,
                    args.batch_size,
                    args.max_segments,
                )
                cached = {
                    "identity": identity,
                    "source": source,
                    "end_seconds": end,
                    "predicted_notes": [asdict(note) for note in predicted],
                }
                if args.cache_notes:
                    atomic_json(cache_path, cached)
            predicted = [Note(**note) for note in cached["predicted_notes"]]
            reference = [
                note
                for note in load_annotation(row["notes_file"])["notes"]
                if note.onset < cached["end_seconds"]
            ]
            score = note_f1(reference, predicted)
            record = {
                "position": position,
                "audio_file": row["mix_audio_file"],
                "subset": row.get("subset"),
                "reference_notes": len(reference),
                "predicted_notes": len(predicted),
                "note_f1": score if math.isfinite(score) else None,
            }
            records.append(record)
            print(
                f"{name} {position + 1}/{len(rows)} note_f1={record['note_f1']}",
                flush=True,
            )
            atomic_json(output / "tracks" / name / f"{position:05d}.json", record)
        scores = [
            record["note_f1"] for record in records if record["note_f1"] is not None
        ]
        summary = {
            "tracks": len(records),
            "scored_tracks": len(scores),
            "mean_note_f1": float(np.mean(scores)) if scores else None,
        }
        if spec.get("subset_field"):
            summary["subsets"] = {}
            for subset in sorted({str(record["subset"]) for record in records}):
                subset_scores = [
                    record["note_f1"]
                    for record in records
                    if str(record["subset"]) == subset and record["note_f1"] is not None
                ]
                summary["subsets"][subset] = {
                    "tracks": sum(
                        str(record["subset"]) == subset for record in records
                    ),
                    "mean_note_f1": float(np.mean(subset_scores))
                    if subset_scores
                    else None,
                }
        results[name] = summary
        atomic_json(output / "summary.json", {"identity": identity, "results": results})
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Resolved training config; defaults to checkpoint sibling",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=Path(__file__).parent / "configs/benchmark/rwc_maps_multtipop.yaml",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, help="Maximum tracks per dataset or RWC subset"
    )
    parser.add_argument(
        "--max-segments", type=int, help="Maximum audio segments per track"
    )
    parser.add_argument("--cache-notes", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
