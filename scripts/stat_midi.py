"""Build corruption marginals from a preloaded MIDI source (no rendering needed)."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from data.datasets import read_indexes
from data.sources import MidiSource
from data.taxonomy import program_to_family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="slakh")
    parser.add_argument("--split", default="train")
    parser.add_argument("--duration", type=float, default=8.192)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.samples <= 0:
        parser.error("duration and samples must be positive")
    index_path = (
        Path(args.root)
        / "yourmt3_indexes"
        / f"{args.dataset}_{args.split}_file_list.json"
    )
    entries = read_indexes(
        args.root, [{"name": args.dataset, "splits": [args.split]}]
    )
    source = MidiSource(entries, [1.0])
    rng = np.random.default_rng(args.seed)
    counts = {name: Counter() for name in ("pitch", "velocity", "duration", "onset")}
    programs = Counter()
    for _ in range(args.samples):
        for note in source.sample(rng, args.duration)["notes"]:
            if note.is_drum or program_to_family(note.program) is None:
                continue
            programs[note.program] += 1
            for name, value in {
                "pitch": note.pitch,
                "velocity": note.velocity,
                "duration": round(note.offset - note.onset, 3),
                "onset": round(note.onset, 3),
            }.items():
                counts[name][value] += 1
    distributions = {}
    for name, counter in counts.items():
        total = sum(counter.values())
        if not total:
            raise ValueError(
                "no renderable melodic notes found in sampled MIDI windows"
            )
        values = sorted(counter)
        distributions[name] = {
            "values": values,
            "counts": [counter[v] for v in values],
            "probs": [counter[v] / total for v in values],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "dataset": args.dataset,
                    "split": args.split,
                    "seed": args.seed,
                    "samples": args.samples,
                    "index_file": index_path.name,
                    "index_sha256": hashlib.sha256(
                        index_path.read_bytes()
                    ).hexdigest(),
                    "midi_parser": {
                        "name": "symusic",
                        "version": version("symusic"),
                    },
                    "clip_policy": "bounded MIDI windows",
                    "time_bin_seconds": 0.001,
                },
                "sampling": {
                    "include_drums": False,
                    "clip_duration_seconds": args.duration,
                },
                "program_note_counts": dict(sorted(programs.items())),
                "distributions": distributions,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
