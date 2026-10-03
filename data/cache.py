"""One rank-local audio cache with explicit producer partitions for cross-augmentation."""

from collections import deque
import logging
import signal
import gc
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from .datasets import identity


def _terminate_worker(signum, frame):
    # Unwind an in-flight renderer's try/finally, which kills and reaps its
    # process group. A default SIGTERM would leave that separate group orphaned.
    raise SystemExit(128 + signum)


def initialize_producer_worker(worker_id):
    signal.signal(signal.SIGTERM, _terminate_worker)


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
        self.producers = producers
        self.options = {
            name: {**config, **(producer_options or {}).get(name, {})}
            for name in producers
        }
        self.loaders, self.iterators = {}, {}
        try:
            for name in producers:
                if self.options[name].get("timeout_retries", 2) < 0:
                    raise ValueError("timeout_retries must be nonnegative")
                self._start_producer(name)
                self.fill(name, self.capacities[name])
        except BaseException:
            self.close()
            raise

    def _start_producer(self, name):
        dataset, options = self.producers[name], self.options[name]
        workers = options["producer_workers"]
        self.loaders[name] = DataLoader(
            dataset,
            batch_size=None,
            sampler=range(self.produced_items[name], len(dataset)),
            collate_fn=identity,
            num_workers=workers,
            worker_init_fn=initialize_producer_worker,
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

    def _stop_producer(self, name):
        iterator = self.iterators.get(name)
        workers = list(getattr(iterator, "_workers", ()) or ())
        for worker in workers:
            if worker.is_alive():
                try:
                    # Give the worker's signal handler a moment to terminate
                    # its renderer process group before using a hard kill.
                    os.kill(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for worker in workers:
            worker.join(timeout=2)
            if worker.is_alive():
                try:
                    os.kill(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                worker.join(timeout=1)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                logging.exception("error while finalizing timed-out producer %s", name)
        self.iterators[name] = None
        self.loaders[name] = None
        gc.collect()
        time.sleep(0.05)

    def fill(self, name, count):
        timeouts = 0
        for _ in range(count):
            while not self.pending[name]:
                try:
                    produced = next(self.iterators[name])
                except RuntimeError as error:
                    # Do not hide exceptions raised inside the dataset itself.
                    if not str(error).startswith("DataLoader timed out after"):
                        raise
                    timeouts += 1
                    retries = self.options[name].get("timeout_retries", 2)
                    if timeouts > retries:
                        raise RuntimeError(
                            f"producer {name} timed out after {timeouts} attempts"
                        ) from error
                    logging.warning(
                        "Producer %s timed out at item %d; restarting workers (%d/%d)",
                        name,
                        self.produced_items[name],
                        timeouts,
                        retries,
                    )
                    self._stop_producer(name)
                    self.pending[name].clear()
                    # Drop the blocked item, not the entire stream. Deterministic
                    # producers would otherwise retry the same bad seed forever.
                    self.produced_items[name] += 1
                    self._start_producer(name)
                    continue
                self.pending[name].extend(produced)
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
        for name in self.iterators:
            self._stop_producer(name)
