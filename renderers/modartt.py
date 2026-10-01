"""Measured preset selection; both Modartt engines use the same error handling."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np

from data.taxonomy import pianoteq_render_group
from .process import UnrenderableSample, render_notes


def pitch_mapping(preset, pitch, is_drum=False):
    measurements = preset.get("drum" if is_drum else "melodic", {}).get("pitches", [])
    audible = [
        row
        for row in measurements
        if row["audible"]
        and (
            row["input_pitch"] == pitch
            if is_drum
            else row.get("detected_midi_mean") is not None
            and np.isfinite(row["detected_midi_mean"])
            and abs(row["detected_midi_mean"] - pitch) < 0.5
        )
    ]
    if not audible:
        return None
    return min(
        audible,
        key=lambda row: (
            0 if is_drum else abs(row["detected_midi_mean"] - pitch),
            abs(row["input_pitch"] - pitch),
        ),
    )["input_pitch"]


class ModarttRenderer:
    def __init__(self, config, sample_rate):
        self.config, self.sample_rate = config, sample_rate
        if not shutil.which(config["binary"]):
            raise FileNotFoundError(
                f"renderer executable not found: {config['binary']}"
            )
        catalog = json.loads(Path(config["capabilities"]).read_text())
        if catalog["schema_version"] != 1:
            raise ValueError("unsupported capability artifact")
        self.presets = catalog["presets"]

    def render(self, score, rng):
        grouped = {}
        for note in score["notes"]:
            grouped.setdefault(128 if note.is_drum else note.program, []).append(note)
        stems, labels, trace = {}, [], []
        for program, notes in grouped.items():
            group = pianoteq_render_group(program)
            candidates = [
                p
                for p in self.presets
                if p["group"] == group
                and (program == 128 or p["source_program"] == program)
                and any(pitch_mapping(p, n.pitch, n.is_drum) is not None for n in notes)
            ]
            if not candidates:
                continue
            preset = candidates[int(rng.integers(len(candidates)))]
            render, targets = [], []
            for note in notes:
                pitch = pitch_mapping(preset, note.pitch, note.is_drum)
                if pitch is not None:
                    render.append(replace(note, pitch=pitch))
                    targets.append(note)
            waveform = render_notes(
                render,
                score["duration"],
                preset["preset"],
                self.config,
                self.sample_rate,
            )
            stems[program] = waveform
            labels.extend(targets)
            trace.append(
                {"program": program, "preset": preset["preset"], "notes": len(targets)}
            )
        if not stems:
            raise UnrenderableSample("score has no matching audible measured presets")
        programs = sorted(stems)
        audio = np.stack([stems[p] for p in programs])
        peak = np.max(np.abs(audio.sum(axis=0)), initial=0)
        if peak > self.config["mix_peak_limit"]:
            audio *= self.config["mix_peak_limit"] / peak
        return {
            "audio": audio,
            "component_programs": [[p] for p in programs],
            "notes": labels,
            "program_labels": {p: {p} for p in programs if p != 128},
            "trace": trace,
        }

    def piano_fallback(self, score, rng):
        """Fallback changes both timbre and target program, never audio alone."""
        score = {
            **score,
            "notes": [replace(n, program=0, is_drum=False) for n in score["notes"]],
        }
        result = self.render(score, rng)
        result["trace"].append({"fallback": "piano"})
        return result
