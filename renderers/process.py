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


def run_command(
    command, timeout_seconds, retries, retry_wait_seconds=1.0, output_path=None
):
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if retries < 0 or not np.isfinite(retry_wait_seconds) or retry_wait_seconds < 0:
        raise ValueError("retries and retry_wait_seconds must be nonnegative")
    for attempt in range(retries + 1):
        # Never accept a previous attempt's partial output as a new success.
        if output_path is not None:
            Path(output_path).unlink(missing_ok=True)
        # Each attempt starts a fresh engine, including after a trial-engine stop.
        child_env = os.environ.copy()
        runtime_lib = child_env.get("PIANOTEQ_RUNTIME_LIB")
        if runtime_lib:
            existing = child_env.get("LD_LIBRARY_PATH", "")
            child_env["LD_LIBRARY_PATH"] = runtime_lib + (
                os.pathsep + existing if existing else ""
            )
        child_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=child_env,
            )
        except OSError as error:
            message = f"could not launch renderer: {type(error).__name__}: {error}"
        else:
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except BaseException as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    # The group may already be gone; still reap our child.
                    if process.poll() is None:
                        process.kill()
                stdout, stderr = process.communicate()
                if not isinstance(error, subprocess.TimeoutExpired):
                    raise
                message = (
                    f"renderer timed out after {timeout_seconds}s: {stderr or stdout}"
                )
            else:
                if process.returncode == 0:
                    return
                message = f"renderer exited {process.returncode}: {stderr or stdout}"
                if "out-of-control[" in message:
                    raise UnrenderableSample(message)
        logging.warning("%s (attempt %d/%d)", message, attempt + 1, retries + 1)
        if attempt < retries:
            time.sleep(retry_wait_seconds * (attempt + 1))
    raise RuntimeError(message)


def render_notes(notes, duration, preset, config, sample_rate):
    """Render full note context, then return the requested crop at sample_rate."""
    if not notes:
        raise UnrenderableSample("empty render request")
    if not np.isfinite(duration) or duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample rate must be positive")
    pre_roll = max(0.0, -min(n.onset for n in notes))
    native_rate = config["render_sample_rate"]
    if native_rate <= 0:
        raise ValueError("render_sample_rate must be positive")
    score = Score(480, ttype="second")
    tracks = {}
    for note in notes:
        if (
            not np.isfinite([note.onset, note.offset]).all()
            or note.offset <= note.onset
        ):
            raise ValueError("invalid rendered note interval")
        if not 0 <= note.pitch <= 127 or not 1 <= note.velocity <= 127:
            raise ValueError("invalid rendered note pitch or velocity")
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
            kept, elapsed = [], 0
            for message in track:
                elapsed += message.time
                if message.type != "program_change":
                    kept.append(message.copy(time=elapsed))
                    elapsed = 0
            if elapsed:
                kept.append(mido.MetaMessage("end_of_track", time=elapsed))
            track[:] = kept
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
        run_command(
            command,
            config["timeout_seconds"],
            config["retries"],
            config.get("retry_wait_seconds", 1.0),
            output_path=wav,
        )
        if not wav.is_file():
            raise RuntimeError(
                "renderer reported success but did not create a WAV file"
            )
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
