# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""event_codec.py: Encodes and decodes events to/from indices

🚀 Improvements:

• Encoding uses a precomputed dictionary in Python. This achieves a time
 complexity of O(1).
• Decoding has time complexity of O(1), while the original code from MT3
 (Gardner et al.) has a time complexity of O(n).

In practice, the performance of this optimized code was 4x faster for encoding
 and decoding compared to the original code.

"""

from typing import List, Dict, Tuple, Optional
from tokenization.note_event_dataclasses import Event, EventRange
# from bisect import bisect_right


class FastCodec:
    """Fast Encoding and decoding Event."""

    def __init__(
        self,
        special_tokens: List[str],
        max_shift_steps: int,
        event_ranges: List[EventRange],
        program_vocabulary: Optional[Dict] = None,
        drum_vocabulary: Optional[Dict] = None,
        extra_tokens: List[str] = [],
        name: Optional[str] = None,
    ):
        """Initializes the FastCodec object.

        :param special_tokens: List of special tokens to include in the vocabulary.
        :param max_shift_steps: The maximum number of steps to shift.
        :param event_ranges: List of EventRange objects.
        :param instr_vocabulary: A dictionary of instrument groups. Please see config/vocabulary.py
            We apply vocabulary only for encoding in training.
        :param drum_vocabulary: A dictionary of drum mapping. Please see config/vocabulary.py
            We apply vocabulary only for encoding in training.

        :param name: Name of the codec.
        """
        # Store the special tokens and event ranges.
        self.special_tokens = special_tokens
        self._special_token_ranges = []
        self._extra_token_ranges = []

        for token in special_tokens:
            self._special_token_ranges.append(EventRange(token, 0, 0))
        for token in extra_tokens:
            self._extra_token_ranges.append(EventRange(token, 0, 0))
        self._shift_range = EventRange(
            type="shift", min_value=0, max_value=max_shift_steps - 1
        )
        self._event_ranges = (
            self._special_token_ranges
            + [self._shift_range]
            + event_ranges
            + self._extra_token_ranges
        )
        # Ensure all event types have unique names.
        assert len(self._event_ranges) == len(
            set([er.type for er in self._event_ranges])
        )

        # Store the name of the codec, so that we can identify it in tokenizer.
        self._name = name

        # Create dictionary for decoding
        self._decode_dict = {}
        self._encode_dict = {}
        self._event_type_range_dict = {}
        idx = 0
        for er in self._event_ranges:
            start_idx = idx
            for value in range(er.min_value, er.max_value + 1):
                self._decode_dict[idx] = Event(type=er.type, value=value)
                self._encode_dict[(er.type, value)] = idx
                idx += 1
            end_idx = idx - 1
            self._event_type_range_dict[er.type] = (start_idx, end_idx)

        self._num_classes = idx

        # Create inverse vocabulary for instrument groups
        if program_vocabulary is not None:
            self.inverse_vocab_program = {}
            self._create_inverse_vocab_program(program_vocabulary)
        else:
            self.inverse_vocab_program = None

        # Create inverse vocabulary for drum mapping
        if drum_vocabulary is not None:
            self.inverse_vocab_drum = {}
            self._create_inverse_vocab_drum(drum_vocabulary)
        else:
            self.inverse_vocab_drum = None

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _create_inverse_vocab_program(self, vocab):
        for key, values in vocab.items():
            for value in values:
                self.inverse_vocab_program[value] = values[0]

    def _create_inverse_vocab_drum(self, vocab):
        for key, values in vocab.items():
            for value in values:
                self.inverse_vocab_drum[value] = values[0]

    def encode_event(self, event: Event) -> int:
        """Encode an event to an index."""
        if (event.type, event.value) not in self._encode_dict:
            raise ValueError(
                f"Unknown event type: {event.type} or value: {event.value}"
            )

        if event.type == "program" and self.inverse_vocab_program is not None:
            # If the event value is not in the vocabulary, use the original value
            _event_value = self.inverse_vocab_program.get(event.value, event.value)
            return self._encode_dict[(event.type, _event_value)]
        elif event.type == "drum" and self.inverse_vocab_drum is not None:
            _event_value = self.inverse_vocab_drum.get(event.value, event.value)
            return self._encode_dict[(event.type, _event_value)]
        else:
            return self._encode_dict[(event.type, event.value)]

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Return [min_id, max_id] for an event type."""
        if event_type not in self._event_type_range_dict:
            raise ValueError(f"Unknown event type: {event_type}")

        return self._event_type_range_dict[event_type]

    def decode_event_index(self, index: int) -> Event:
        """Decode an event index to an Event."""
        if index < 0 or index >= self.num_classes:
            raise ValueError(f"Unknown event index: {index}")
        decoded_event = self._decode_dict[index]

        # Create a new event with the same type and value
        return Event(type=decoded_event.type, value=decoded_event.value)
