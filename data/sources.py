"""Preloaded MIDI symbols. Audio and rendered-clip caching belong elsewhere."""

import logging
from pathlib import Path

import numpy as np
import torch
from symusic import Score

from tokenization.note_event_dataclasses import Note


NOTE_DTYPE = np.dtype(
    [
        ("onset", "<f4"),
        ("offset", "<f4"),
        ("pitch", "u1"),
        ("velocity", "u1"),
        ("program", "u1"),
        ("is_drum", "?"),
    ]
)


class MidiSource:
    def __init__(self, entries, dataset_probabilities, sample_rate=16000):
        """Preload each MIDI once; keep duplicate index rows and their weights.

        Symbols live in one read-only shared tensor so spawn workers do not
        deserialize a separate corpus each. There is no eviction or refresh.
        """
        self.sample_rate = sample_rate
        self.records, probabilities, packed = [], [], []
        loaded, cursor = {}, 0
        for (dataset, rows), probability in zip(entries.items(), dataset_probabilities):
            for row in rows:
                path = str(Path(row["midi_file"]).resolve())
                if path not in loaded:
                    score = Score(path, ttype="second")
                    values = []
                    for track in score.tracks:
                        for note in track.notes:
                            onset = max(0.0, float(note.time))
                            values.append(
                                (
                                    onset,
                                    max(float(note.end), onset + 0.001),
                                    int(note.pitch),
                                    max(1, int(note.velocity)),
                                    128 if track.is_drum else int(track.program),
                                    bool(track.is_drum),
                                )
                            )
                    values.sort()
                    notes = np.asarray(values, dtype=NOTE_DTYPE)
                    loaded[path] = (cursor, cursor + len(notes))
                    cursor += len(notes)
                    packed.append(notes)
                first, last = loaded[path]
                duration = row["n_frames"] / row.get("sample_rate", sample_rate)
                self.records.append((dataset, path, float(duration), first, last))
                probabilities.append(float(probability) / len(rows))
        if not self.records:
            raise ValueError("MidiSource needs at least one indexed MIDI")
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.probabilities /= self.probabilities.sum()
        self.storage = torch.from_numpy(
            np.concatenate(packed).view(np.uint8)
        ).share_memory_()
        self.preloaded_bytes = self.storage.numel()
        logging.info(
            "MidiSource: %d unique files, %.1f MiB symbols",
            len(loaded),
            self.preloaded_bytes / 2**20,
        )

    def sample(self, rng, duration):
        index = int(rng.choice(len(self.records), p=self.probabilities))
        dataset, path, full_duration, first, last = self.records[index]
        maximum = max(
            0,
            round(full_duration * self.sample_rate)
            - round(duration * self.sample_rate),
        )
        start = int(rng.integers(maximum + 1)) / self.sample_rate
        symbols = self.storage.numpy().view(NOTE_DTYPE)[first:last]
        selected = symbols[
            (symbols["offset"] > start) & (symbols["onset"] < start + duration)
        ]
        # Preserve the original combined-MIDI source's bounded crop semantics.
        notes = [
            Note(
                bool(n["is_drum"]),
                int(n["program"]),
                max(0.0, float(n["onset"]) - start),
                min(duration, float(n["offset"]) - start),
                int(n["pitch"]),
                int(n["velocity"]),
            )
            for n in selected
        ]
        return {
            "notes": notes,
            "duration": duration,
            "source_id": path,
            "source_dataset": dataset,
            "start": start,
        }
