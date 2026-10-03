"""NSynth index and one shared rotating waveform bank. Metadata stays a dict."""

from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch


class NSynthBank:
    def __init__(
        self,
        root,
        sample_rate=16000,
        notes_per_bucket=8,
        size_gib=None,
        init_workers=8,
        seed=0,
    ):
        if sample_rate <= 0 or notes_per_bucket < 0 or init_workers <= 0:
            raise ValueError("invalid NSynth bank sample rate or capacity")
        if size_gib is not None and (not np.isfinite(size_gib) or size_gib <= 0):
            raise ValueError("NSynth bank size_gib must be positive or null")
        root = Path(root)
        self.directory = next(
            (
                p
                for p in (
                    root / "train/data",
                    root / "train/audio",
                    root / "nsynth-train/audio",
                    root / "audio",
                )
                if p.is_dir()
            ),
            None,
        )
        if self.directory is None:
            raise FileNotFoundError(f"NSynth train audio not found under {root}")
        self.sample_rate, self.frames = sample_rate, 4 * sample_rate
        self.sources, self.buckets = [], {}
        for path in sorted(self.directory.glob("*.wav")):
            instrument, pitch, velocity = path.stem.rsplit("-", 2)
            family, kind, _ = instrument.rsplit("_", 2)
            source = {
                "key": path.stem,
                "family": family,
                "kind": kind,
                "pitch": int(pitch),
                "velocity": int(velocity),
            }
            self.buckets.setdefault((family, int(pitch)), []).append(len(self.sources))
            self.sources.append(source)
        if not self.sources:
            raise ValueError("NSynth bank is empty")
        self.pitches = {
            family: sorted(p for f, p in self.buckets if f == family)
            for family, _ in self.buckets
        }
        self.rng = np.random.default_rng(seed)
        self.lock = mp.get_context("spawn").Lock()
        self.owner_pid = os.getpid()
        self.audio = None
        self.rows, self.orders, self.cursors = {}, {}, {}
        self.refresh_count, self.refresh_cursor = 0, 0
        capacities = {
            key: min(notes_per_bucket, len(ids)) for key, ids in self.buckets.items()
        }
        if size_gib is not None:
            budget = int(size_gib * 2**30) // (self.frames * 4)
            if budget < len(self.buckets):
                raise ValueError(
                    "NSynth bank budget must hold at least one source per pitch bucket"
                )
            capacities = {key: 1 for key in self.buckets}
            remaining = min(budget, len(self.sources)) - len(capacities)
            while remaining:
                for key, ids in self.buckets.items():
                    if capacities[key] < len(ids) and remaining:
                        capacities[key] += 1
                        remaining -= 1
        initial = []
        for key, ids in self.buckets.items():
            capacity = capacities[key]
            order = self.rng.permutation(ids).tolist()
            self.orders[key], self.cursors[key] = order, capacity
            self.rows[key] = list(range(len(initial), len(initial) + capacity))
            initial.extend(order[:capacity])
        self.cached_source_count = len(initial)
        self.cache_bytes = len(initial) * self.frames * 4
        if initial:
            available = os.statvfs("/dev/shm")
            if self.cache_bytes > available.f_bavail * available.f_frsize:
                raise MemoryError(
                    f"NSynth bank needs {self.cache_bytes / 2**30:.2f} GiB shared memory; reduce bank.size_gib/notes_per_bucket"
                )
            logging.info(
                "NSynth bank: loading %d waveforms (%.2f GiB)",
                len(initial),
                self.cache_bytes / 2**30,
            )
            self.audio = torch.empty(
                (len(initial), self.frames), dtype=torch.float32
            ).share_memory_()
            self.ids = torch.tensor(initial).share_memory_()
            self.drawn = torch.zeros(len(initial), dtype=torch.bool).share_memory_()
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=init_workers) as pool:
                for row, waveform in enumerate(pool.map(self.read, initial)):
                    self.audio[row].copy_(torch.from_numpy(waveform))
                    if (row + 1) % 5000 == 0 or row + 1 == len(initial):
                        logging.info(
                            "NSynth bank: loaded %d/%d waveforms in %.1fs",
                            row + 1,
                            len(initial),
                            time.monotonic() - started,
                        )

    def read(self, source_id):
        source = self.sources[source_id]
        waveform, rate = sf.read(
            self.directory / (source["key"] + ".wav"), dtype="float32", always_2d=True
        )
        waveform = waveform.mean(axis=1)
        if rate != self.sample_rate:
            waveform = soxr.resample(waveform, rate, self.sample_rate, quality="HQ")
        output = np.zeros(self.frames, dtype=np.float32)
        output[: min(len(waveform), self.frames)] = waveform[: self.frames]
        return output

    def draw(self, family, pitch, velocity, rng, exclude):
        # Shuffle equal-distance pitch candidates, then prefer the nearest velocity.
        pitches = self.pitches.get(family, ())
        distances = sorted({abs(p - pitch) for p in pitches})
        with self.lock:
            for distance in distances:
                candidates = [p for p in pitches if abs(p - pitch) == distance]
                for actual_pitch in rng.permutation(candidates):
                    key = family, int(actual_pitch)
                    ids = (
                        self.buckets[key]
                        if self.audio is None
                        else [int(self.ids[row]) for row in self.rows[key]]
                    )
                    ids = [i for i in ids if self.sources[i]["key"] not in exclude]
                    if not ids:
                        continue
                    delta = min(
                        abs(self.sources[i]["velocity"] - velocity) for i in ids
                    )
                    ids = [
                        i
                        for i in ids
                        if abs(self.sources[i]["velocity"] - velocity) == delta
                    ]
                    selected = ids[int(rng.integers(len(ids)))]
                    if self.audio is not None:
                        row = next(
                            r for r in self.rows[key] if int(self.ids[r]) == selected
                        )
                        waveform = self.audio[row].numpy().copy()
                        self.drawn[row] = True
                    else:
                        waveform = None
                    break
                else:
                    continue
                break
            else:
                raise LookupError(f"no unused NSynth source for {family}/{pitch}")
        return self.sources[selected], self.read(
            selected
        ) if waveform is None else waveform

    def refresh(self, count):
        """Only the owner replaces drawn slots; waveform and identity swap together."""
        if self.audio is None or count == 0:
            return
        if os.getpid() != self.owner_pid:
            raise RuntimeError("only the bank owner may refresh sources")
        keys = list(self.buckets)
        replaced, examined = 0, 0
        max_scan = len(keys) * max(1, max(map(len, self.rows.values())))
        while replaced < count and examined < max_scan:
            key = keys[self.refresh_cursor % len(keys)]
            self.refresh_cursor += 1
            examined += 1
            with self.lock:
                rows = self.rows[key]
                row = next((r for r in rows if bool(self.drawn[r])), None)
                resident = {int(self.ids[r]) for r in rows}
                if row is None or len(resident) == len(self.buckets[key]):
                    continue
                while True:
                    cursor = self.cursors[key]
                    if cursor == len(self.orders[key]):
                        self.orders[key] = self.rng.permutation(
                            self.buckets[key]
                        ).tolist()
                        cursor = 0
                    selected = self.orders[key][cursor]
                    self.cursors[key] = cursor + 1
                    if selected not in resident:
                        break
            waveform = self.read(selected)
            with self.lock:
                self.audio[row].copy_(torch.from_numpy(waveform))
                self.ids[row] = selected
                self.drawn[row] = False
            replaced += 1
        self.refresh_count += replaced
