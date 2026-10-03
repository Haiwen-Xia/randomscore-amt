"""Measured preset selection; both Modartt engines use the same error handling."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import logging

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
        self.excluded_presets = set(config.get("excluded_presets", []))

    def render(self, score, rng):
        grouped = {}
        for note in score["notes"]:
            grouped.setdefault(128 if note.is_drum else note.program, []).append(note)
        stems, labels, trace, failures = {}, [], [], []
        for program, notes in grouped.items():
            group = pianoteq_render_group(program)
            candidates = [
                p
                for p in self.presets
                if p["group"] == group
                and p["preset"] not in self.excluded_presets
                and (program == 128 or p["source_program"] == program)
                and any(pitch_mapping(p, n.pitch, n.is_drum) is not None for n in notes)
            ]
            if not candidates:
                trace.append(
                    {
                        "program": program,
                        "status": "skipped",
                        "reason": "no_matching_measured_preset",
                        "notes": len(notes),
                    }
                )
                continue
            preset = candidates[int(rng.integers(len(candidates)))]
            render, targets = [], []
            for note in notes:
                pitch = pitch_mapping(preset, note.pitch, note.is_drum)
                if pitch is not None:
                    render.append(replace(note, pitch=pitch))
                    targets.append(note)
            try:
                waveform = render_notes(
                    render,
                    score["duration"],
                    preset["preset"],
                    self.config,
                    self.sample_rate,
                )
                if (
                    waveform.shape != (round(score["duration"] * self.sample_rate),)
                    or not np.isfinite(waveform).all()
                ):
                    raise RuntimeError(
                        "renderer returned invalid audio shape or values"
                    )
                if np.max(np.abs(waveform), initial=0) < self.config["minimum_peak"]:
                    raise UnrenderableSample("near-silent audio")
            except Exception as error:
                # Match the old capability renderer: a failed stem must not
                # discard valid stems. Only successful notes become targets.
                failures.append(error)
                trace.append(
                    {
                        "program": program,
                        "preset": preset["preset"],
                        "status": "failed",
                        "sample_retryable": isinstance(error, UnrenderableSample),
                        "error": f"{type(error).__name__}: {error}"[-2000:],
                    }
                )
                logging.warning(
                    "Dropping failed stem program=%s preset=%s: %s",
                    program,
                    preset["preset"],
                    error,
                )
                continue
            stems[program] = waveform
            labels.extend(targets)
            trace.append(
                {
                    "program": program,
                    "preset": preset["preset"],
                    "notes": len(targets),
                    "status": "rendered",
                }
            )
        if not stems:
            fatal = [
                error for error in failures if not isinstance(error, UnrenderableSample)
            ]
            if fatal:
                raise RuntimeError(
                    f"all Modartt stems failed; first fatal error: {fatal[0]}"
                ) from fatal[0]
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
