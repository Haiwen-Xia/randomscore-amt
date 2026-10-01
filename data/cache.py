"""One rank-local audio cache with explicit producer partitions for cross-augmentation."""

from collections import deque
import numpy as np
from torch.utils.data import DataLoader
from .datasets import identity


def allocate_counts(total, weights):
    if (
        not weights
        or not np.isfinite(list(weights.values())).all()
        or any(w <= 0 for w in weights.values())
    ):
        raise ValueError("partition weights must be finite and positive")
    total_weight = sum(weights.values())
    counts = {
        name: int(round(total * weight / total_weight))
        for name, weight in weights.items()
    }
    if sum(counts.values()) != total or any(
        not np.isclose(counts[name], total * weight / total_weight) or counts[name] <= 0
        for name, weight in weights.items()
    ):
        raise ValueError(
            f"cannot divide {total} clips exactly among partition weights {weights}"
        )
    return counts


class ClipCache:
    def __init__(
        self, producers, config, seed, start_indices=None, producer_options=None
    ):
        self.rng = np.random.default_rng(seed)
        self.weights = config.get("partition_weights") or {
            name: 1 for name in producers
        }
        if set(self.weights) != set(producers):
            raise ValueError("partition weights must match producers")
        self.capacities = allocate_counts(config["capacity"], self.weights)
        self.refresh_counts = allocate_counts(config["refresh_clips"], self.weights)
        # Older native offline checkpoints stored one producer cursor.
        if isinstance(start_indices, int):
            start_indices = {"offline": start_indices}
        self.produced_items = {
            name: (start_indices or {}).get(name, 0) for name in producers
        }
        self.clips = {name: [] for name in producers}
        self.updated = {name: set() for name in producers}
        self.write_positions = {name: 0 for name in producers}
        self.pending = {name: deque() for name in producers}
        self.loaders, self.iterators = {}, {}
        try:
            for name, dataset in producers.items():
                options = {**config, **(producer_options or {}).get(name, {})}
                workers = options["workers"]
                self.loaders[name] = DataLoader(
                    dataset,
                    batch_size=None,
                    sampler=range(self.produced_items[name], len(dataset)),
                    collate_fn=identity,
                    num_workers=workers,
                    **(
                        {
                            "prefetch_factor": 1,
                            "persistent_workers": True,
                            "multiprocessing_context": "spawn",
                            "timeout": options["timeout_seconds"],
                        }
                        if workers
                        else {}
                    ),
                )
                self.iterators[name] = iter(self.loaders[name])
                self.fill(name, self.capacities[name])
        except BaseException:
            self.close()
            raise

    def fill(self, name, count):
        for _ in range(count):
            if not self.pending[name]:
                self.pending[name].extend(next(self.iterators[name]))
                self.produced_items[name] += 1
                if not self.pending[name]:
                    raise RuntimeError(f"producer {name} returned no clips")
            clip = self.pending[name].popleft()
            slot = self.write_positions[name]
            if slot < len(self.clips[name]):
                self.clips[name][slot] = clip
            else:
                self.clips[name].append(clip)
            self.updated[name].add(slot)
            self.write_positions[name] = (slot + 1) % self.capacities[name]

    def refill(self):
        for name, count in self.refresh_counts.items():
            self.fill(name, count)

    def sample(self, count):
        samples = []
        for name, number in allocate_counts(count, self.weights).items():
            if number > len(self.clips[name]):
                raise ValueError("batch size exceeds cache partition capacity")
            fresh = sorted(self.updated[name])
            self.rng.shuffle(fresh)
            slots = fresh[:number]
            self.updated[name].difference_update(slots)
            if len(slots) < number:
                remaining = sorted(set(range(len(self.clips[name]))) - set(slots))
                slots.extend(
                    self.rng.choice(
                        remaining, number - len(slots), replace=False
                    ).tolist()
                )
            samples.extend(self.clips[name][i] for i in slots)
        self.rng.shuffle(samples)
        return samples

    def donors(self, count, exclude):
        # Donors span all partitions; final audio need not retain the base quota.
        sources = {}
        for clips in self.clips.values():
            for clip in clips:
                if clip["source_id"] not in exclude:
                    sources.setdefault(clip["source_id"], []).append(clip)
        names = list(sources)
        if count > len(names):
            raise RuntimeError(
                f"cache has {len(names)} eligible donor sources; {count} requested"
            )
        selected = self.rng.choice(len(names), count, replace=False)
        return [
            sources[names[i]][int(self.rng.integers(len(sources[names[i]])))]
            for i in selected
        ]

    def close(self):
        for iterator in self.iterators.values():
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
