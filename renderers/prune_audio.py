from __future__ import annotations

from collections.abc import Iterable

import numpy as np


DEFAULTS = {
    "sample_dur": 4.0,
    "note_off_sec": 3.0,
    "release_extra_sec": 0.05,
    "analysis_frame_sec": 0.05,
    "analysis_hop_sec": 0.01,
    "attack_abs": 1e-4,
    "attack_rel": 0.08,
    "attack_search_sec": 0.60,
    "rms_use_3s_probe": True,
    "rms_3s_probe_half_window_sec": 0.08,
    "rms_3s_probe_rel": 0.12,
    "rms_3s_probe_abs": 0.01,
    "fade_in_sec": 0.002,
    "fade_out_sec": 0.008,
}


def _validate_config(config: dict, sample_rate: int) -> None:
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if (
        not np.isfinite(config["sample_dur"])
        or not 0.0 < config["note_off_sec"] < config["sample_dur"]
    ):
        raise ValueError("note_off_sec must be inside finite sample_dur")
    for name in (
        "analysis_frame_sec",
        "analysis_hop_sec",
        "attack_search_sec",
        "rms_3s_probe_half_window_sec",
    ):
        value = float(config[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "release_extra_sec",
        "attack_abs",
        "attack_rel",
        "rms_3s_probe_rel",
        "rms_3s_probe_abs",
        "fade_in_sec",
        "fade_out_sec",
    ):
        value = float(config[name])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def _fix_length_batch(audio_batch: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros((audio_batch.shape[0], size), dtype=np.float32)
    take = min(audio_batch.shape[1], size)
    if take > 0:
        output[:, :take] = audio_batch[:, :take]
    return output


def _rms_envelope_batch(
    audio_batch: np.ndarray, frame_length: int, hop_length: int
) -> np.ndarray:
    source = audio_batch.astype(np.float32, copy=False)
    if source.shape[1] < frame_length:
        source = _fix_length_batch(source, frame_length)
    frame_count = 1 + (source.shape[1] - frame_length) // hop_length
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    squared = np.square(source, dtype=np.float64)
    prefix = np.empty((source.shape[0], source.shape[1] + 1), dtype=np.float64)
    prefix[:, 0] = 0.0
    np.cumsum(squared, axis=1, out=prefix[:, 1:])
    energy = prefix[:, starts + frame_length] - prefix[:, starts]
    return np.sqrt(energy / float(frame_length) + 1e-12).astype(np.float32)


def _analyze_row(
    audio: np.ndarray,
    envelope: np.ndarray,
    sample_rate: int,
    config: dict,
    frame_length: int,
    hop_length: int,
) -> dict:
    note_off = min(int(round(config["note_off_sec"] * sample_rate)), audio.size)
    probe_center = int(round(config["note_off_sec"] * sample_rate / hop_length))
    probe_half = max(
        1, int(round(config["rms_3s_probe_half_window_sec"] * sample_rate / hop_length))
    )
    probe_left = max(0, probe_center - probe_half)
    probe_right = min(envelope.size, probe_center + probe_half + 1)
    peak = float(np.max(envelope, initial=0.0)) + 1e-12
    probe_threshold = max(config["rms_3s_probe_abs"], config["rms_3s_probe_rel"] * peak)
    probe_rms = (
        float(np.mean(envelope[probe_left:probe_right]))
        if probe_right > probe_left
        else 0.0
    )
    use_3s = bool(
        config["rms_use_3s_probe"]
        and probe_right > probe_left
        and probe_rms <= probe_threshold
    )
    source_end = note_off if use_3s else audio.size
    note_frame_end = min(
        envelope.size,
        max(1, 1 + (max(note_off, frame_length) - frame_length) // hop_length),
    )
    note_envelope = envelope[:note_frame_end]
    peak = float(np.max(note_envelope, initial=0.0))
    threshold = max(config["attack_abs"], config["attack_rel"] * peak)
    active = np.flatnonzero(note_envelope >= threshold)
    attack_frame = int(active[0]) if active.size else 0
    attack_search_frames = max(
        1, int(round(config["attack_search_sec"] * sample_rate / hop_length))
    )
    peak_search_end = min(note_frame_end, attack_frame + attack_search_frames + 1)
    attack_end_frame = attack_frame + int(
        np.argmax(note_envelope[attack_frame:peak_search_end])
    )
    attack_start = min(attack_frame * hop_length, max(source_end - 1, 0))
    attack_end = min(attack_end_frame * hop_length, max(source_end - 1, 0))
    return dict(
        attack_start=attack_start,
        attack_end=max(attack_end, attack_start),
        note_off=note_off,
        source_end=source_end,
        use_3s=use_3s,
        probe_rms=probe_rms,
        probe_threshold=float(probe_threshold),
    )


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: dict = DEFAULTS,
) -> dict:
    """Detect onset and the usable source boundary in one waveform."""

    analyses, _ = _analyze_batch([audio], sample_rate, config)
    return analyses[0]


def _analyze_batch(
    audio_list: list[np.ndarray],
    sample_rate: int,
    config: dict,
) -> tuple[list[dict], np.ndarray]:
    _validate_config(config, sample_rate)
    sample_length = int(round(config["sample_dur"] * sample_rate))
    source = np.zeros((len(audio_list), sample_length), dtype=np.float32)
    for row_index, audio in enumerate(audio_list):
        row = np.asarray(audio, dtype=np.float32).reshape(-1)
        take = min(row.size, sample_length)
        source[row_index, :take] = row[:take]

    frame_length = max(1, int(round(config["analysis_frame_sec"] * sample_rate)))
    hop_length = max(1, int(round(config["analysis_hop_sec"] * sample_rate)))
    envelope = _rms_envelope_batch(source, frame_length, hop_length)
    analyses = [
        _analyze_row(
            source[row_index],
            envelope[row_index],
            sample_rate,
            config,
            frame_length,
            hop_length,
        )
        for row_index in range(source.shape[0])
    ]
    return analyses, source


def _render_row(
    source: np.ndarray,
    duration: float,
    analysis: dict,
    sample_rate: int,
    config: dict,
    normalize_target_peak: float | None,
) -> np.ndarray:
    note_length = max(0, int(round(float(duration) * sample_rate)))
    release_length = max(0, int(round(config["release_extra_sec"] * sample_rate)))
    output_length = note_length + release_length
    output = np.zeros(output_length, dtype=np.float32)
    source_take = min(
        output_length, max(analysis["source_end"] - analysis["attack_start"], 0)
    )
    if source_take == 0:
        return output
    output[:source_take] = source[
        analysis["attack_start"] : analysis["attack_start"] + source_take
    ]

    if release_length and note_length < source_take:
        release_take = source_take - note_length
        phase = np.linspace(0.0, np.pi / 2.0, release_length, dtype=np.float32)
        output[note_length:source_take] *= np.cos(phase[:release_take]) ** 2
        if source_take == output_length:
            output[source_take - 1] = 0.0

    fade_in = min(source_take, int(round(config["fade_in_sec"] * sample_rate)))
    if fade_in > 1:
        phase = np.linspace(0.0, np.pi / 2.0, fade_in, dtype=np.float32)
        output[:fade_in] *= np.sin(phase)
    fade_out = min(source_take, int(round(config["fade_out_sec"] * sample_rate)))
    if fade_out > 0:
        phase = np.linspace(0.0, np.pi / 2.0, fade_out, dtype=np.float32)
        output[source_take - fade_out : source_take] *= np.cos(phase) ** 2
        output[source_take - 1] = 0.0

    if normalize_target_peak is not None:
        peak = float(np.max(np.abs(output), initial=0.0))
        if peak > 1e-9:
            output *= np.float32(float(normalize_target_peak) / peak)
    return output


def prune_audio(
    audio: np.ndarray,
    duration: float,
    sample_rate: int,
    config: dict = DEFAULTS,
    normalize_target_peak: float | None = None,
) -> np.ndarray:
    """Render contiguous source audio with onset alignment and a synthetic release."""

    return prune_audio_batch(
        [audio],
        [duration],
        sample_rate,
        config,
        normalize_target_peak=normalize_target_peak,
    )[0]


def prune_audio_batch(
    audio_list: Iterable[np.ndarray],
    durations: Iterable[float],
    sample_rate: int,
    config: dict,
    normalize_target_peak: float | None = None,
) -> list[np.ndarray]:
    """Analyze a source batch once, then assemble each requested duration."""

    audios = [np.asarray(audio, dtype=np.float32) for audio in audio_list]
    duration_array = np.asarray(list(durations), dtype=np.float64)
    if not audios:
        raise ValueError("audio_list must not be empty")
    if duration_array.shape != (len(audios),):
        raise ValueError("durations must contain one value per audio")
    if not np.all(np.isfinite(duration_array)) or np.any(duration_array <= 0.0):
        raise ValueError("durations must be finite and positive")
    if normalize_target_peak is not None and not 0.0 < normalize_target_peak <= 1.0:
        raise ValueError("normalize_target_peak must be in (0, 1]")

    unique_audios: list[np.ndarray] = []
    unique_by_identity: dict[int, int] = {}
    source_rows: list[int] = []
    for audio in audios:
        identity = id(audio)
        row_index = unique_by_identity.get(identity)
        if row_index is None:
            row_index = len(unique_audios)
            unique_by_identity[identity] = row_index
            unique_audios.append(audio)
        source_rows.append(row_index)

    analyses, source = _analyze_batch(unique_audios, sample_rate, config)
    return [
        _render_row(
            source[source_rows[output_index]],
            float(duration_array[output_index]),
            analyses[source_rows[output_index]],
            sample_rate,
            config,
            normalize_target_peak,
        )
        for output_index in range(len(audios))
    ]
