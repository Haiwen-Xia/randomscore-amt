"""Generic sampler/renderer composition. All samples and renderer outputs are dicts."""

import logging

import numpy as np
from torch.utils.data import Dataset

from .datasets import events_for_clip
from renderers.process import UnrenderableSample


class AudioMidiSampleBuilder:
    def __init__(self, sample_rate, input_frames, segment_seconds):
        self.sample_rate, self.input_frames = sample_rate, input_frames
        self.stride = round(segment_seconds * sample_rate)
        if not 0 < input_frames <= self.stride:
            raise ValueError("input_frames must fit in the rendered segment stride")

    def build(self, score, rendered, route_name, sample_seed):
        """Use rendered labels and crop stems/notes together, including boundary TIEs."""
        audio = rendered["audio"]
        groups = rendered["component_programs"]
        if (
            audio.ndim != 2
            or audio.shape[0] != len(groups)
            or any(not group for group in groups)
        ):
            raise ValueError(
                "renderer must provide program identities for every audio component"
            )
        separable = all(len(group) == 1 for group in groups)
        programs = (
            [group[0] for group in groups]
            if separable
            else sorted({p for group in groups for p in group})
        )
        if not separable:
            # One NSynth family stem may contain multiple fine programs. Preserve
            # their canonical tokens, but do not claim those programs are separable.
            audio = audio.sum(axis=0, keepdims=True)
        if not np.isfinite(audio).all():
            raise ValueError("renderer returned nonfinite audio")
        frames = round(score["duration"] * self.sample_rate)
        if audio.shape[1] != frames or frames % self.stride:
            raise ValueError(
                "render duration must contain an integer number of training segments"
            )
        samples = []
        for start in range(0, frames, self.stride):
            events, ties = events_for_clip(
                rendered["notes"],
                start / self.sample_rate,
                self.input_frames / self.sample_rate,
                program_labels=rendered["program_labels"],
            )
            samples.append(
                {
                    "audio": audio[:, start : start + self.input_frames].copy(),
                    "programs": np.asarray(programs, dtype=np.int64),
                    "has_stems": separable,
                    "has_unannotated": False,
                    "note_events": events,
                    "tie_note_events": ties,
                    "source_id": f"{route_name}:{sample_seed}",
                    "dataset": route_name,
                    "trace": {
                        "midi_source": score["source_id"],
                        "midi_start": score["start"],
                        "segment_start": start / self.sample_rate,
                        "sampler": score.get("trace", {}),
                        "renderer": rendered.get("trace", []),
                    },
                }
            )
        return samples


class SeededRenderedDataset(Dataset):
    def __init__(self, routes, sample_builder, seed, max_attempts=32):
        """Each named route is {sampler, renderer, weight, optional fallback}."""
        self.routes, self.sample_builder, self.seed = routes, sample_builder, seed
        self.names = list(routes)
        weights = np.asarray([r["weight"] for r in routes.values()], dtype=float)
        if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("online routes need finite positive weights")
        self.probabilities = weights / weights.sum()
        self.max_attempts = max_attempts

    def __len__(self):
        return 2**31

    def __getitem__(self, index):
        seed = int(
            np.random.SeedSequence([self.seed, int(index)]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        return self.sample_from_seed(seed)

    def sample_from_seed(self, seed, route_name=None):
        rng = np.random.default_rng(seed)
        if route_name is None:
            route_name = self.names[
                int(rng.choice(len(self.names), p=self.probabilities))
            ]
        route = self.routes[route_name]
        last_score = None
        for attempt in range(self.max_attempts):
            score = route["sampler"].sample(rng)
            try:
                if not score["notes"]:
                    raise UnrenderableSample("empty symbolic score")
                last_score = score
                rendered = route["renderer"].render(score, rng)
            except UnrenderableSample as error:
                last_error = error
                continue
            if attempt:
                logging.info(
                    "%s: sample %s succeeded after %d rejections",
                    route_name,
                    seed,
                    attempt,
                )
            return self.sample_builder.build(score, rendered, route_name, seed)
        if route.get("fallback") is not None and last_score is not None:
            logging.warning(
                "%s: sample %s uses fallback after %d rejections",
                route_name,
                seed,
                self.max_attempts,
            )
            rendered = route["fallback"](last_score, rng)
            return self.sample_builder.build(last_score, rendered, route_name, seed)
        raise UnrenderableSample(
            f"{route_name} failed after {self.max_attempts} attempts: {last_error}"
        )
