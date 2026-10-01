"""Fixed 13-channel note targets, including coarse/partial program labels."""

from typing import Any
from copy import deepcopy
import logging
import numpy as np
from .note_event_dataclasses import Event
from .tokenizer import NoteEventTokenizer
from .vocabulary import MT3_FULL_PLUS, GM_DRUM_NOTES, PROGRAM_TO_CHANNEL, NUM_CHANNELS


def _constraint_for_event(event: Any, kind: str) -> frozenset[int]:
    constraint = event.label_constraint
    if constraint is None:
        value = int(event.pitch if kind == "drum_pitch" else event.program)
        return frozenset({value})
    if constraint["kind"] != kind:
        raise ValueError(f"event requires {kind} supervision, got {constraint['kind']}")
    return frozenset(int(value) for value in constraint["allowed_values"])


def _event_constraints(
    note_events: list[Any], tie_note_events: list[Any], start_time: float
) -> tuple[list[Any], dict[int, frozenset[int]]]:
    from tokenization.note2event import note_event2event

    events = note_event2event(note_events, tie_note_events, start_time, sort=True)
    constrained: dict[int, frozenset[int]] = {}
    program_state: int | None = None
    active_program_event: int | None = None

    event_cursor = 0
    for tie in tie_note_events:
        if int(tie.program) != program_state:
            while events[event_cursor].type != "program":
                event_cursor += 1
            active_program_event = event_cursor
            constrained[event_cursor] = _constraint_for_event(tie, "program")
            program_state = int(tie.program)
            event_cursor += 1
        elif active_program_event is not None:
            constrained[active_program_event] &= _constraint_for_event(tie, "program")

    for note_event in note_events:
        if bool(note_event.is_drum) and int(note_event.velocity) == 0:
            continue
        if bool(note_event.is_drum):
            while event_cursor < len(events) and events[event_cursor].type != "drum":
                event_cursor += 1
            if event_cursor >= len(events):
                raise ValueError("could not align a drum event with tokenization")
            constrained[event_cursor] = _constraint_for_event(note_event, "drum_pitch")
            event_cursor += 1
            continue
        if int(note_event.program) != program_state:
            while event_cursor < len(events) and events[event_cursor].type != "program":
                event_cursor += 1
            if event_cursor >= len(events):
                raise ValueError("could not align a program event with tokenization")
            active_program_event = event_cursor
            constrained[event_cursor] = _constraint_for_event(note_event, "program")
            program_state = int(note_event.program)
            event_cursor += 1
        elif active_program_event is not None:
            constrained[active_program_event] &= _constraint_for_event(
                note_event, "program"
            )

    if any(not allowed for allowed in constrained.values()):
        raise ValueError("notes sharing one emitted program token have disjoint labels")
    return events, constrained


class TargetTokenizer:
    def __init__(self, max_length=256, max_shift_steps=206):
        self.max_length = max_length
        self.tokenizer = NoteEventTokenizer(
            max_length=max_length,
            max_shift_steps=max_shift_steps,
            program_vocabulary=MT3_FULL_PLUS,
            drum_vocabulary=GM_DRUM_NOTES,
        )
        self.num_tokens = self.tokenizer.num_tokens
        self._warned_unsupported = set()

    def encode(self, samples):
        """Samples contain relative note_events and tie_note_events; PAD=0, EOS=1."""
        tokens = np.zeros((len(samples), NUM_CHANNELS, self.max_length), dtype=np.int64)
        allowed_ids = np.full((*tokens.shape, 16), -1, dtype=np.int64)
        codec = self.tokenizer.codec
        for row, sample in enumerate(samples):
            channels = [([], []) for _ in range(NUM_CHANNELS)]
            for kind, key in enumerate(("note_events", "tie_note_events")):
                for event in deepcopy(sample[key]):
                    program = 128 if event.is_drum else event.program
                    if program not in PROGRAM_TO_CHANNEL:
                        if program not in self._warned_unsupported:
                            logging.warning(
                                "program %s has no decoder channel; dropping its events from supervision",
                                program,
                            )
                            self._warned_unsupported.add(program)
                        continue
                    channels[PROGRAM_TO_CHANNEL[program]][kind].append(event)
            for channel, (notes, ties) in enumerate(channels):
                notes.sort(
                    key=lambda n: (n.time, n.is_drum, n.program, n.velocity, n.pitch)
                )
                ties.sort(key=lambda n: (n.program, n.pitch))
                events, constraints = _event_constraints(notes, ties, 0.0)
                encoded = [codec.encode_event(e) for e in events] + [1]
                encoded = encoded[: self.max_length]
                tokens[row, channel, : len(encoded)] = encoded
                for position, values in constraints.items():
                    if position >= self.max_length:
                        continue
                    event = events[position]
                    if event.type == "program" and any(
                        PROGRAM_TO_CHANNEL[v] != channel for v in values
                    ):
                        raise ValueError(
                            "partial program labels must stay within one decoder channel"
                        )
                    ids = sorted(
                        {codec.encode_event(Event(event.type, int(v))) for v in values}
                    )
                    if ids == [int(tokens[row, channel, position])]:
                        continue
                    if len(ids) > allowed_ids.shape[-1]:
                        raise ValueError("too many alternative labels")
                    allowed_ids[row, channel, position, : len(ids)] = ids
        return tokens, allowed_ids
