"""Read prepared YourMT3 indexes. Dataset combinations are ordinary config lists."""

import json
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from torch.utils.data import Dataset

from tokenization.note_event_dataclasses import Note, NoteEvent


class AnnotationUnpickler(pickle.Unpickler):
    """Read existing annotation files without importing the project that wrote them."""

    def find_class(self, module, name):
        if module in {"utils.note_event_dataclasses", "note_event_dataclasses"}:
            return {"Note": Note, "NoteEvent": NoteEvent}[name]
        return super().find_class(module, name)


def load_annotation(path):
    # Prepared local .npy dictionaries contain Python note objects. Like np.load
    # with allow_pickle=True, this reader is for trusted dataset artifacts only.
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            _, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            _, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported annotation NPY version {version}")
        if not dtype.hasobject:
            raise ValueError(f"expected a prepared annotation dictionary: {path}")
        return AnnotationUnpickler(handle).load().item()


def read_indexes(root, datasets):
    """Return one list per dataset; repeated splits deliberately retain multiplicity."""
    root = Path(root)
    result = {}
    for spec in datasets:
        entries = []
        for split in spec["splits"]:
            path = root / "yourmt3_indexes" / f"{spec['name']}_{split}_file_list.json"
            rows = json.loads(path.read_text())
            for key in sorted(rows, key=int):
                row = dict(rows[key])
                # Index paths may be absolute on the machine that prepared them.
                # Relocation is relative to data root, not to the old codebase.
                for field, value in row.items():
                    if field.endswith("_file") and value:
                        original = Path(value)
                        if not original.is_absolute():
                            row[field] = str(root / original)
                        elif not original.exists():
                            parts = original.parts
                            folder = next(
                                (
                                    i
                                    for i, p in enumerate(parts)
                                    if p.endswith("_yourmt3_16k") or p == spec["name"]
                                ),
                                None,
                            )
                            if folder is not None:
                                row[field] = str(root.joinpath(*parts[folder:]))
                entries.append(row)
        if not entries or spec["name"] in result:
            raise ValueError(f"empty or duplicate dataset {spec['name']}")
        result[spec["name"]] = entries
    return result


def events_for_clip(notes, start, duration, allowed_programs=None, program_labels=None):
    """Keep negative onsets as TIEs; do not invent offsets at crop boundaries."""
    events, ties = [], []
    for note in notes:
        onset, offset = note.onset - start, note.offset - start
        if offset <= 0 or onset >= duration:
            continue
        program = 128 if note.is_drum else int(note.program)
        values = (
            {note.pitch}
            if note.is_drum
            else set(
                program_labels[program]
                if program_labels is not None
                else allowed_programs or [program]
            )
        )
        constraint = {
            "kind": "drum_pitch" if note.is_drum else "program",
            "allowed_values": frozenset(values),
        }
        if onset < 0 and not note.is_drum:
            ties.append(
                NoteEvent(
                    False, program, None, 1, note.pitch, label_constraint=constraint
                )
            )
        elif onset >= 0:
            events.append(
                NoteEvent(
                    note.is_drum,
                    program,
                    onset,
                    1,
                    note.pitch,
                    label_constraint=constraint,
                )
            )
        if not note.is_drum and 0 < offset < duration:
            events.append(
                NoteEvent(
                    False, program, offset, 0, note.pitch, label_constraint=constraint
                )
            )
    events.sort(key=lambda n: (n.time, n.is_drum, n.program, n.velocity, n.pitch))
    ties.sort(key=lambda n: (n.program, n.pitch))
    return events, ties


def read_mix(entry, sample_rate):
    audio, source_rate = sf.read(
        entry["mix_audio_file"], dtype="float32", always_2d=True
    )
    audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        audio = resample_poly(
            audio, sample_rate // divisor, source_rate // divisor
        ).astype(np.float32)
    return audio


def read_segments(entry, spec, starts, frames, sample_rate):
    """Read each source once, returning independently cropped stem-aware samples."""
    notes = load_annotation(entry["notes_file"])["notes"]
    stem = (
        load_annotation(entry["stem_file"])
        if spec.get("stems") and entry.get("stem_file")
        else None
    )
    metadata = stem if stem is not None else entry
    programs = np.asarray(metadata["program"], dtype=np.int64)
    drums = np.asarray(metadata.get("is_drum", programs == 128), dtype=bool)
    programs = np.where(drums, 128, programs)
    annotated = programs[programs != 129]
    # A repeated program does not identify a unique stem in the note annotation.
    separable = stem is not None and len(annotated) == len(set(annotated))
    if stem is not None:
        full_audio = np.asarray(stem["audio_array"], dtype=np.float32)
        separable &= len(programs) == full_audio.shape[0]
        crops = [full_audio[:, s : s + frames] for s in starts]
    else:
        crops = []
        with sf.SoundFile(entry["mix_audio_file"]) as handle:
            if handle.samplerate != sample_rate:
                raise ValueError(
                    "training indexes must reference audio at the configured sample rate"
                )
            for start in starts:
                handle.seek(min(start, len(handle)))
                crops.append(
                    handle.read(frames, dtype="float32", always_2d=True).mean(axis=1)[
                        None
                    ]
                )
    samples = []
    for start, audio in zip(starts, crops):
        audio = np.pad(audio, ((0, 0), (0, max(0, frames - audio.shape[-1]))))
        events, ties = events_for_clip(
            notes,
            start / sample_rate,
            frames / sample_rate,
            spec.get("allowed_programs"),
        )
        active = {event.program for event in events + ties}
        has_stems = bool(separable and active.issubset(set(annotated)))
        selected_programs = programs.copy()
        if has_stems:
            limit = spec.get("max_components")
            if limit and len(programs) > limit:
                energy = np.square(audio, dtype=np.float64).mean(axis=1)
                selected = sorted(
                    sorted(
                        range(len(programs)),
                        key=lambda i: (
                            energy[i] > 0,
                            programs[i] in active,
                            energy[i],
                            -i,
                        ),
                        reverse=True,
                    )[:limit]
                )
                audio, selected_programs = audio[selected], programs[selected]
                events = [e for e in events if e.program in selected_programs]
                ties = [e for e in ties if e.program in selected_programs]
        else:
            audio = audio.sum(axis=0, keepdims=True)
            selected_programs = np.asarray(
                sorted(active or set(annotated)), dtype=np.int64
            )
        unannotated = (
            bool(129 in selected_programs) if has_stems else bool(129 in programs)
        )
        if unannotated and 129 not in selected_programs:
            selected_programs = np.append(selected_programs, 129)
        samples.append(
            {
                "audio": np.array(audio, dtype=np.float32, copy=True),
                "programs": selected_programs,
                "has_stems": has_stems,
                "has_unannotated": unannotated,
                "note_events": events,
                "tie_note_events": ties,
                "source_id": str(entry["notes_file"]),
                "dataset": spec["name"],
                "trace": {
                    "kind": "offline",
                    "audio_file": str(entry["mix_audio_file"]),
                    "notes_file": str(entry["notes_file"]),
                    "start": start / sample_rate,
                },
            }
        )
    return samples


class OfflineDataset(Dataset):
    """A producer item samples one file, then reads several random crops from it."""

    def __init__(self, root, datasets, frames, sample_rate, segments_per_source, seed):
        self.specs = datasets
        self.entries = read_indexes(root, datasets)
        self.frames, self.sample_rate = frames, sample_rate
        self.segments_per_source, self.seed = segments_per_source, seed
        counts = np.asarray(
            [len(self.entries[s["name"]]) for s in datasets], dtype=float
        )
        # YourMT3 assigned every file in dataset i: weight_i * (1 - n_i / N).
        # Thus the total dataset probability includes n_i, not just weight_i.
        masses = (
            np.ones(1)
            if len(counts) == 1
            else counts
            * (1 - counts / counts.sum())
            * np.asarray([s["weight"] for s in datasets])
        )
        if not np.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError("dataset sampling weights must be finite and positive")
        self.probabilities = masses / masses.sum()

    def __len__(self):
        return 2**31

    def __getitem__(self, index):
        # Independent of worker scheduling and worker count.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        spec = self.specs[int(rng.choice(len(self.specs), p=self.probabilities))]
        entries = self.entries[spec["name"]]
        entry = entries[int(rng.integers(len(entries)))]
        maximum = max(0, entry["n_frames"] - self.frames)
        starts = rng.integers(maximum + 1, size=self.segments_per_source).tolist()
        return read_segments(entry, spec, starts, self.frames, self.sample_rate)


def identity(value):
    """DataLoader batch_size=None: leave variable-length notes as Python objects."""
    return value
