"""Render notes from NSynth recordings; return the labels of what actually sounded."""

from dataclasses import replace

import numpy as np

from data.taxonomy import FAMILY_PROGRAMS, program_to_family
from .prune_audio import DEFAULTS, prune_audio_batch
from .process import UnrenderableSample


class NSynthRenderer:
    def __init__(self, bank, config):
        self.bank, self.config = bank, config
        self.prune = {**DEFAULTS, **config.get("prune", {})}
        if any(
            not 0 < config[name] <= 1
            for name in ("note_target_peak", "mix_target_peak")
        ):
            raise ValueError("NSynth target peaks must be in (0, 1]")
        if config["boundary_context_seconds"] < 0 or config["max_source_attempts"] <= 0:
            raise ValueError("invalid NSynth boundary context or retry limit")

    def render(self, score, rng):
        sr, duration = self.bank.sample_rate, score["duration"]
        frames = round(duration * sr)
        used, jobs = set(), []
        for note in score["notes"]:
            family = program_to_family(note.program)
            if (
                note.is_drum
                or family is None
                or note.offset <= 0
                or note.onset >= duration
            ):
                continue
            pre = min(max(-note.onset, 0), self.config["boundary_context_seconds"])
            post = min(
                max(note.offset - duration, 0), self.config["boundary_context_seconds"]
            )
            length = pre + min(note.offset, duration) - max(note.onset, 0) + post
            try:
                source, audio = self.bank.draw(
                    family, note.pitch, note.velocity, rng, used
                )
            except LookupError:
                continue
            used.add(source["key"])
            jobs.append((note, family, source, audio, pre, length))
        if not jobs:
            raise UnrenderableSample("score has no available NSynth notes")
        pruned = prune_audio_batch(
            [j[3] for j in jobs],
            [j[5] for j in jobs],
            sr,
            self.prune,
            normalize_target_peak=self.config["note_target_peak"],
        )
        stems, labels, trace = {}, [], []
        component_programs, program_labels = {}, {}
        for (note, family, source, audio, pre, length), waveform in zip(jobs, pruned):
            first = round(pre * sr)
            start = max(0, round(note.onset * sr))
            count = min(
                frames - start,
                round(
                    (
                        min(note.offset + self.prune["release_extra_sec"], duration)
                        - max(note.onset, 0)
                    )
                    * sr
                ),
            )
            rejected = set(used)
            audible = waveform[first : first + count]
            for attempt in range(self.config["max_source_attempts"]):
                if np.max(np.abs(audible), initial=0) > 1e-9:
                    break
                if attempt + 1 == self.config["max_source_attempts"]:
                    break
                rejected.add(source["key"])
                try:
                    source, audio = self.bank.draw(
                        family, note.pitch, note.velocity, rng, rejected
                    )
                except LookupError:
                    break
                used.add(source["key"])
                waveform = prune_audio_batch(
                    [audio],
                    [length],
                    sr,
                    self.prune,
                    normalize_target_peak=self.config["note_target_peak"],
                )[0]
                audible = waveform[first : first + count]
            if not np.max(np.abs(audible), initial=0) > 1e-9:
                trace.append({"status": "skipped_silent", "pitch": note.pitch})
                continue
            # Keep the sampled fine program as the canonical teacher-forcing
            # token. Only Synth Pad is remapped to NSynth's Synth Lead channel.
            program = 80 if 88 <= note.program <= 95 else note.program
            stem = stems.setdefault(family, np.zeros(frames, dtype=np.float32))
            component_programs.setdefault(family, set()).add(program)
            program_labels[program] = FAMILY_PROGRAMS[family]
            stem[start : start + len(audible)] += audible
            labels.append(replace(note, program=program, pitch=source["pitch"]))
            trace.append(
                {
                    "source_key": source["key"],
                    "requested_pitch": note.pitch,
                    "pitch": source["pitch"],
                }
            )
        if not labels:
            raise UnrenderableSample("all NSynth notes rendered silence")
        families = sorted(stems)
        audio = np.stack([stems[f] for f in families])
        peak = np.max(np.abs(audio.sum(axis=0)), initial=0)
        if peak > 1e-9:
            audio *= self.config["mix_target_peak"] / peak
        return {
            "audio": audio,
            "notes": labels,
            "component_programs": [sorted(component_programs[f]) for f in families],
            "program_labels": program_labels,
            "trace": trace,
        }
