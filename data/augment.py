"""YourMT3 intra-stem and cache-based cross-stem mixing, with one sample format."""

from copy import deepcopy

import numpy as np

from tokenization.note2event import note_event2event


def select_programs(sample, programs):
    mask = np.isin(sample["programs"], programs)
    sample["audio"] = sample["audio"][mask]
    sample["programs"] = sample["programs"][mask]
    for key in ("note_events", "tie_note_events"):
        sample[key] = [event for event in sample[key] if event.program in programs]
    sample["has_unannotated"] = bool(129 in programs)
    return sample


def intra_augment(sample, keep_probability, rng):
    sample = deepcopy(sample)
    if not sample["has_stems"]:
        return sample
    programs = sample["programs"]
    if sample["has_unannotated"]:
        if rng.random() < keep_probability:
            select_programs(sample, programs[programs != 129])
        return sample
    active = np.asarray(
        sorted({e.program for e in sample["note_events"] + sample["tie_note_events"]}),
        dtype=np.int64,
    )
    kept = active[rng.random(len(active)) < keep_probability]
    if len(active) and not len(kept):
        kept = rng.choice(active, 1)
    return select_programs(sample, kept)


def mix_audio(audio, rng, amp_range, normalize=True):
    gains = rng.uniform(*amp_range, size=(audio.shape[0], 1)).astype(np.float32)
    mixed = (audio * gains).sum(axis=0)
    if normalize:
        mixed /= np.max(np.abs(mixed), initial=0.0) + 1e-7
    return mixed


def regroup_audio(samples, max_stems, rng):
    """Merge equal program units, then randomly combine units to bound stem count."""
    groups = {}
    for sample in samples:
        audio, programs = sample["audio"], sample["programs"]
        if audio.shape[0] > 1:
            for program, stem in zip(programs, audio):
                groups.setdefault((int(program),), []).append(stem)
        elif audio.shape[0]:
            groups.setdefault(tuple(sorted(programs)), []).append(audio[0])
    groups = list(groups.values())
    while len(groups) > max_stems:
        a, b = rng.choice(len(groups), 2, replace=False)
        groups[a].extend(groups[b])
        del groups[b]
    frames = samples[0]["audio"].shape[-1]
    result = np.zeros((len(groups), frames), dtype=np.float32)
    for index, group in enumerate(groups):
        result[index] = (
            group[0]
            if len(group) == 1
            else mix_audio(np.stack(group), rng, [0.9, 1.0], False)
        )
    return result


def within_event_budget(samples, donor, max_events=1024):
    # This is a pre-mix guard; per-channel token truncation is done later.
    if sum(len(s["note_events"]) for s in [*samples, donor]) < max_events // 3:
        return True
    return (
        sum(
            len(note_event2event(s["note_events"], s["tie_note_events"], 0.0))
            for s in [*samples, donor]
        )
        < max_events
    )


def source_metadata(sample):
    """Return compact, JSON-ready provenance for an accepted mix component."""
    return {
        "source_id": sample["source_id"],
        "dataset": sample["dataset"],
        "programs": [int(program) for program in sample["programs"]],
        "trace": sample.get("trace"),
    }


def augment_batch(samples, cache, config, rng):
    max_k, tau, alpha = config["max_k"], config["tau"], config["alpha"]
    survival = np.exp(-np.power(np.arange(max_k + 1) * tau, alpha))
    probabilities = -np.diff(np.append(survival, 0.0))
    counts = rng.choice(max_k + 1, len(samples), p=probabilities)
    exclude = {s["source_id"] for s in samples}
    result = []
    for sample, count in zip(samples, counts):
        base = intra_augment(sample, config["stem_keep_probability"], rng)
        gathered = [base]
        # An unannotated mixture cannot acquire valid negative labels by mixing.
        if not base["has_unannotated"]:
            donors = cache.donors(int(count), exclude)
            for donor in donors:
                donor = intra_augment(donor, config["stem_keep_probability"], rng)
                if donor["has_unannotated"]:
                    continue
                used = {p for s in gathered for p in s["programs"]}
                unique = [p for p in donor["programs"] if p not in used]
                if len(unique) != len(donor["programs"]):
                    if not unique or not donor["has_stems"]:
                        continue
                    # Check the unfiltered donor, matching the original guard.
                    if not within_event_budget(gathered, donor):
                        break
                    donor = select_programs(donor, unique)
                elif not within_event_budget(gathered, donor):
                    break
                gathered.append(donor)
        use_regroup = (count > 0 and not base["has_unannotated"]) or base[
            "audio"
        ].shape[0] > config["max_stems"]
        stems = (
            regroup_audio(gathered, config["max_stems"], rng)
            if use_regroup
            else base["audio"]
        )
        mixed = dict(base)
        mixed["audio"] = mix_audio(stems, rng, config["amplitude_range"])[None]
        mixed["programs"] = np.unique(np.concatenate([s["programs"] for s in gathered]))
        mixed["has_stems"] = False
        mixed["source_ids"] = [s["source_id"] for s in gathered]
        mixed["sources"] = [source_metadata(s) for s in gathered]
        for key in ("note_events", "tie_note_events"):
            mixed[key] = [e for s in gathered for e in s[key]]
        result.append(mixed)
    return result
