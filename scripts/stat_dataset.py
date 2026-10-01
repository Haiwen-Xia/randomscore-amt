"""Inspect the configured offline mixture: python -m scripts.stat_dataset."""

import json
from collections import Counter

import hydra
from omegaconf import OmegaConf

from data.datasets import OfflineDataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    dataset = OfflineDataset(
        config["data_root"],
        config["data"]["train"],
        config["audio"]["input_frames"],
        config["audio"]["sample_rate"],
        config["cache"]["segments_per_source"],
        config["run"]["seed"],
    )
    report = {}
    for spec, probability in zip(dataset.specs, dataset.probabilities):
        entries = dataset.entries[spec["name"]]
        programs = Counter(p for e in entries for p in set(map(int, e["program"])))
        report[spec["name"]] = {
            "splits": spec["splits"],
            "index_entries": len(entries),
            "unique_annotations": len({e["notes_file"] for e in entries}),
            "hours": sum(
                e["n_frames"] / e.get("sample_rate", config["audio"]["sample_rate"])
                for e in entries
            )
            / 3600,
            "config_weight": spec["weight"],
            "sampling_probability": float(probability),
            "files_per_program": dict(sorted(programs.items())),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
