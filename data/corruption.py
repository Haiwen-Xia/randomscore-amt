"""The single symbolic sampler: replace MIDI attributes using measured marginals."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .taxonomy import program_to_family


def draw_strength(value, rng):
    if value == "uniform":
        return float(rng.random())
    if value == "beta_2_1":
        return float(rng.beta(2, 1))
    if value == "beta_5_1":
        return float(rng.beta(5, 1))
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"invalid corruption strength/probability: {value}")
    return float(value)


class CorruptionSampler:
    def __init__(self, source, statistics_path, config, duration):
        self.source, self.config, self.duration = source, config, duration
        if config.get("max_notes") is not None and config["max_notes"] <= 0:
            raise ValueError("max_notes must be positive or null")
        stats = json.loads(Path(statistics_path).read_text())
        if stats["sampling"]["include_drums"]:
            raise ValueError("corruption requires drum-excluded marginal statistics")
        if not np.isclose(stats["sampling"]["clip_duration_seconds"], duration):
            raise ValueError("statistics duration must match render duration")
        self.distributions = {}
        for name in ("pitch", "velocity", "duration", "onset"):
            if name not in stats["distributions"]:
                continue
            distribution = stats["distributions"][name]
            values, probs = (
                np.asarray(distribution["values"]),
                np.asarray(distribution["probs"], dtype=float),
            )
            if name == "duration":
                valid = (values >= config["min_duration"]) & (
                    values <= config["max_duration"]
                )
                values, probs = values[valid], probs[valid]
            if (
                not len(values)
                or not np.isfinite(probs).all()
                or (probs < 0).any()
                or probs.sum() <= 0
            ):
                raise ValueError(f"invalid marginal distribution: {name}")
            self.distributions[name] = (values, probs / probs.sum())
        counts = {
            int(p): count
            for p, count in stats["program_note_counts"].items()
            if program_to_family(int(p))
        }
        probabilities = np.asarray(list(counts.values()), dtype=float)
        if not len(counts) or probabilities.sum() <= 0:
            raise ValueError("statistics have no renderable programs")
        self.distributions["program"] = (
            np.asarray(list(counts)),
            probabilities / probabilities.sum(),
        )
        if set(config["keep_dimensions"]) - {
            "program",
            "pitch",
            "velocity",
            "onset",
            "duration",
        }:
            raise ValueError("unknown corruption dimension")
        for name in ("p", "s"):
            draw_strength(config[name], np.random.default_rng(0))

    def sample(self, rng):
        score = self.source.sample(rng, self.duration)
        probability, strength = (
            draw_strength(self.config["p"], rng),
            draw_strength(self.config["s"], rng),
        )
        keep = self.config["keep_dimensions"]
        notes = [
            n for n in score["notes"] if not n.is_drum and program_to_family(n.program)
        ]
        limit = self.config.get("max_notes")
        if limit is not None and len(notes) > limit:
            notes = [
                notes[i] for i in sorted(rng.choice(len(notes), limit, replace=False))
            ]
        sampled = []
        for note in notes:
            if rng.random() >= probability:
                sampled.append(note)
                continue
            replacement = {}
            for name in ("program", "pitch", "velocity", "duration"):
                values, probs = self.distributions[name]
                replacement[name] = float(rng.choice(values, p=probs))
            if "onset" in self.distributions:
                values, probs = self.distributions["onset"]
                valid = (values < self.duration) & (
                    values + replacement["duration"] > 0
                )
            else:
                valid = np.array([], dtype=bool)
            if valid.any():
                replacement["onset"] = float(
                    rng.choice(values[valid], p=probs[valid] / probs[valid].sum())
                )
            else:
                end = max(
                    self.duration - min(replacement["duration"], self.duration), 0.0
                )
                replacement["onset"] = float(rng.uniform(0, end)) if end else 0.0
            origin = {
                "program": note.program,
                "pitch": note.pitch,
                "velocity": note.velocity,
                "onset": note.onset,
                "duration": note.offset - note.onset,
            }
            values = {
                name: old
                if name in keep
                else replacement[name]
                if name == "program"
                else old + strength * (replacement[name] - old)
                for name, old in origin.items()
            }
            onset, duration = values["onset"], values["duration"]
            if onset + duration <= 0:
                if "duration" in keep:
                    onset = float(np.nextafter(-duration, np.inf))
                else:
                    duration = float(np.nextafter(-onset, np.inf))
            onset = min(onset, float(np.nextafter(self.duration, -np.inf)))
            sampled.append(
                replace(
                    note,
                    onset=onset,
                    offset=onset + max(duration, np.finfo(np.float32).eps),
                    pitch=int(np.clip(np.rint(values["pitch"]), 0, 127)),
                    velocity=int(np.clip(np.rint(values["velocity"]), 1, 127)),
                    program=int(values["program"]),
                )
            )
        # Drop later overlaps of identities that the coarse target cannot represent.
        end_by_identity, retained = {}, []
        for note in sorted(
            sampled,
            key=lambda n: (n.onset, n.offset - n.onset, n.pitch, n.velocity, n.program),
        ):
            identity = program_to_family(note.program), note.pitch
            if note.onset >= end_by_identity.get(identity, -np.inf):
                retained.append(note)
                end_by_identity[identity] = note.offset
        return {
            **score,
            "notes": retained,
            "trace": {
                "p": probability,
                "s": strength,
                "source_notes": len(notes),
                "dropped_overlaps": len(sampled) - len(retained),
            },
        }
