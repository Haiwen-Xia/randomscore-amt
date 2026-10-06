"""Export augmented training samples as WAV, MIDI, and provenance JSON files."""

import json
from collections import defaultdict, deque
from pathlib import Path

import hydra
import mido
import numpy as np
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from data.augment import augment_batch
from data.cache import ClipCache, allocate_counts
from data.datasets import OfflineDataset


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _inspection_notes(sample, duration):
    active = defaultdict(deque)
    notes = []
    for event in sample["tie_note_events"]:
        active[(event.is_drum, event.program, event.pitch)].append(0.0)
    for event in sorted(
        sample["note_events"],
        key=lambda item: (item.time, item.velocity == 0, item.program, item.pitch),
    ):
        key = event.is_drum, event.program, event.pitch
        time = float(event.time)
        if event.is_drum:
            if event.velocity:
                notes.append((*key, time, min(duration, time + 0.05)))
        elif event.velocity:
            active[key].append(time)
        elif active[key]:
            onset = active[key].popleft()
            notes.append((*key, onset, max(time, onset + 0.001)))
    for key, onsets in active.items():
        for onset in onsets:
            notes.append((*key, onset, max(duration, onset + 0.001)))
    return sorted(notes, key=lambda item: (item[3], item[1], item[2], item[4]))


def write_midi(sample, path, duration, ticks_per_beat=480, tempo=500000):
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    midi.tracks.append(tempo_track)
    grouped = defaultdict(list)
    for is_drum, program, pitch, onset, offset in _inspection_notes(sample, duration):
        grouped[(is_drum, program)].append((pitch, onset, offset))
    melodic_channels = [channel for channel in range(16) if channel != 9]
    for index, ((is_drum, program), notes) in enumerate(sorted(grouped.items())):
        channel = 9 if is_drum else melodic_channels[index % len(melodic_channels)]
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=f"program_{program}", time=0))
        if not is_drum:
            track.append(
                mido.Message("program_change", channel=channel, program=program, time=0)
            )
        messages = []
        for pitch, onset, offset in notes:
            messages.extend(
                [
                    (
                        onset,
                        1,
                        mido.Message(
                            "note_on", channel=channel, note=pitch, velocity=100
                        ),
                    ),
                    (
                        offset,
                        0,
                        mido.Message(
                            "note_off", channel=channel, note=pitch, velocity=0
                        ),
                    ),
                ]
            )
        previous_tick = 0
        for time, _, message in sorted(
            messages, key=lambda item: (item[0], item[1])
        ):
            tick = round(mido.second2tick(time, ticks_per_beat, tempo))
            message.time = max(0, tick - previous_tick)
            track.append(message)
            previous_tick = tick
        midi.tracks.append(track)
    midi.save(path)


def sample_metadata(sample, index, sample_rate):
    frames = int(sample["audio"].shape[-1])
    sources = sample.get("sources") or [
        {
            "source_id": sample["source_id"],
            "dataset": sample["dataset"],
            "programs": [int(program) for program in sample["programs"]],
            "trace": sample.get("trace"),
        }
    ]
    sources = [
        {"role": "base" if position == 0 else "donor", **source}
        for position, source in enumerate(sources)
    ]
    return _json_value(
        {
            "schema_version": 1,
            "index": index,
            "dataset": sample["dataset"],
            "sample_rate": sample_rate,
            "frames": frames,
            "duration_seconds": frames / sample_rate,
            "programs": [int(program) for program in sample["programs"]],
            "note_events": len(sample["note_events"]),
            "tie_note_events": len(sample["tie_note_events"]),
            "sources": sources,
        }
    )


def export_samples(samples, output_dir, sample_rate, context=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "count": len(samples),
        **(context or {}),
        "samples": [],
    }
    for index, sample in enumerate(samples):
        stem = f"{index:03d}"
        wav_path, midi_path, json_path = (
            output_dir / f"{stem}.wav",
            output_dir / f"{stem}.mid",
            output_dir / f"{stem}.json",
        )
        audio = np.asarray(sample["audio"], dtype=np.float32).sum(axis=0)
        sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
        duration = len(audio) / sample_rate
        write_midi(sample, midi_path, duration)
        metadata = sample_metadata(sample, index, sample_rate)
        json_path.write_text(json.dumps(metadata, indent=2) + "\n")
        manifest["samples"].append(
            {
                "index": index,
                "audio": wav_path.name,
                "midi": midi_path.name,
                "metadata": json_path.name,
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_samples(config):
    seed = config["run"]["seed"]
    sample_rate = config["audio"]["sample_rate"]
    dataset = OfflineDataset(
        config["data_root"],
        config["data"]["train"],
        config["audio"]["input_frames"],
        sample_rate,
        config["cache"]["segments_per_source"],
        seed,
    )
    producers, producer_options, banks = {"offline": dataset}, {}, []
    if config.get("online") is not None:
        from data.build import build_online
        from data.sources import MidiSource

        sources = {
            "training": MidiSource(
                dataset.entries, dataset.probabilities, sample_rate
            )
        }
        online, banks = build_online(config, sources, seed + 500003)
        producers["online"] = online
        producer_options["online"] = {
            "producer_workers": config["online"]["producer_workers"]
        }
    count = config["inspection"]["count"]
    if count <= 0:
        raise ValueError("inspection.count must be positive")
    weights = config["cache"].get("partition_weights") or {
        name: 1 for name in producers
    }
    allocate_counts(count, weights)
    cache = ClipCache(
        producers, config["cache"], seed, producer_options=producer_options
    )
    try:
        rng = np.random.default_rng(seed)
        return augment_batch(cache.sample(count), cache, config["augmentation"], rng)
    finally:
        cache.close()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    samples = build_samples(config)
    export_samples(
        samples,
        config["inspection"]["output_dir"],
        config["audio"]["sample_rate"],
        {
            "run_name": config["run"]["name"],
            "seed": config["run"]["seed"],
            "offline_datasets": [spec["name"] for spec in config["data"]["train"]],
            "online_routes": list((config.get("online") or {}).get("routes", {})),
        },
    )
    print(
        f"Wrote {len(samples)} inspection pairs "
        f"to {config['inspection']['output_dir']}"
    )


if __name__ == "__main__":
    main()
