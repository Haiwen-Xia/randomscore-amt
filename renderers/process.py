"""Shared subprocess lifecycle for Pianoteq and Organteq."""

import logging
from pathlib import Path
import signal
import subprocess
import os
import time
import tempfile

import numpy as np
import soundfile as sf
import soxr
from symusic import Score, Track, Note


class UnrenderableSample(RuntimeError):
    """A valid sample/preset combination failed; another sample may succeed."""


def run_command(command, timeout_seconds, retries):
    for attempt in range(retries + 1):
        # Each attempt starts a fresh engine, including after a trial-engine stop.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            message = f"renderer timed out after {timeout_seconds}s"
        else:
            if process.returncode == 0:
                return
            message = f"renderer exited {process.returncode}: {stderr or stdout}"
            if "out-of-control[" in message:
                raise UnrenderableSample(message)
        logging.warning("%s (attempt %d/%d)", message, attempt + 1, retries + 1)
        if attempt < retries:
            time.sleep(attempt + 1)
    raise RuntimeError(message)


def render_notes(notes, duration, preset, config, sample_rate):
    """Render full note context, then return the requested crop at sample_rate."""
    if not notes:
        raise UnrenderableSample("empty render request")
    pre_roll = max(0.0, -min(n.onset for n in notes))
    native_rate = config["render_sample_rate"]
    score = Score(480, ttype="second")
    tracks = {}
    for note in notes:
        if (
            not np.isfinite([note.onset, note.offset]).all()
            or note.offset <= note.onset
        ):
            raise ValueError("invalid rendered note interval")
        key = bool(note.is_drum)
        track = tracks.setdefault(key, Track(program=0, is_drum=key, ttype="second"))
        track.notes.append(
            Note(
                time=note.onset + pre_roll,
                duration=note.offset - note.onset,
                pitch=note.pitch,
                velocity=note.velocity,
                ttype="second",
            )
        )
    score.tracks.extend(tracks.values())
    with tempfile.TemporaryDirectory(prefix="modartt-") as directory:
        midi, wav = Path(directory) / "input.mid", Path(directory) / "output.wav"
        score.dump_midi(midi)
        # The selected preset owns timbre; remove the MIDI program changes.
        import mido

        midi_file = mido.MidiFile(midi)
        for track in midi_file.tracks:
            track[:] = [
                message for message in track if message.type != "program_change"
            ]
        midi_file.save(midi)
        command = [
            config["binary"],
            "--headless",
            "--no-prefs",
            "--midi",
            str(midi),
            "--wav",
            str(wav),
            "--preset",
            preset,
        ]
        if native_rate != 44100:
            command.extend(["--rate", str(native_rate)])
        run_command(command, config["timeout_seconds"], config["retries"])
        audio, actual_rate = sf.read(wav, dtype="float32", always_2d=True)
    if actual_rate != native_rate or not np.isfinite(audio).all():
        raise RuntimeError("renderer returned invalid audio or sample rate")
    audio = audio.mean(axis=1)
    first, length = round(pre_roll * native_rate), round(duration * native_rate)
    cropped = np.zeros(length, dtype=np.float32)
    audible = audio[first : first + length]
    cropped[: len(audible)] = audible
    if native_rate != sample_rate:
        cropped = soxr.resample(cropped, native_rate, sample_rate, quality="HQ")
    target = round(duration * sample_rate)
    cropped = np.pad(cropped[:target], (0, max(0, target - len(cropped))))
    if np.max(np.abs(cropped), initial=0) < config["minimum_peak"]:
        raise UnrenderableSample(f"preset {preset} produced silence")
    return cropped.astype(np.float32)
