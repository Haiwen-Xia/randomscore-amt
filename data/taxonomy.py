from __future__ import annotations


FAMILY_PROGRAM: dict[str, int] = {
    "bass": 32,
    "brass": 56,
    "flute": 73,
    "guitar": 24,
    "keyboard": 0,
    "mallet": 11,
    "organ": 19,
    "reed": 71,
    "string": 40,
    "synth_lead": 80,
    # YourMT3 reserves internal program 100 for singing voice.
    "vocal": 100,
}


def program_to_family(program: int) -> str | None:
    """Map raw GM programs represented by NSynth to one coarse family."""

    program = int(program)
    if 0 <= program <= 7:
        return "keyboard"
    if 8 <= program <= 15:
        return "mallet"
    if 16 <= program <= 23:
        return "organ"
    if 24 <= program <= 31:
        return "guitar"
    if 32 <= program <= 39:
        return "bass"
    if 40 <= program <= 55:
        return "string"
    if 56 <= program <= 63:
        return "brass"
    if 64 <= program <= 71:
        return "reed"
    if 72 <= program <= 79:
        return "flute"
    if 80 <= program <= 95:
        return "synth_lead"
    return None


def target_program_to_family(program: int) -> str | None:
    """Map generated target programs, including YourMT3's singing program."""

    program = int(program)
    if program == FAMILY_PROGRAM["vocal"]:
        return "vocal"
    return program_to_family(program)


# Program sets represented by each coarse family under the current repository
# taxonomy. These are semantic training labels, not token IDs.
FAMILY_PROGRAMS: dict[str, frozenset[int]] = {
    family: (
        frozenset({representative})
        if family == "vocal"
        else (
            frozenset(range(80, 88))
            if family == "synth_lead"
            else frozenset(
                program
                for program in range(128)
                if program_to_family(program) == family
            )
        )
    )
    for family, representative in FAMILY_PROGRAM.items()
}


# Renderer-specific component groups. These are intentionally separate from the
# NSynth family taxonomy above: they describe available Pianoteq sound pools,
# while note.program continues to carry the original training target.
PIANOTEQ_GROUP_PROGRAM: dict[str, int] = {
    "piano": 0,
    "electric_piano": 4,
    "guitar": 24,
    "bass": 32,
    "harp": 46,
    "synth": 80,
    "chromatic": 11,
    "drum": 128,
}

PIANOTEQ_RENDER_GROUPS = tuple(PIANOTEQ_GROUP_PROGRAM)


def pianoteq_render_group(program: int, *, is_drum: bool = False) -> str:
    """Map a full-dataset target onto one Pianoteq render group."""

    if is_drum or int(program) == 128:
        return "drum"
    program = int(program)
    if not 0 <= program <= 127:
        raise ValueError(f"program must be in [0, 128], got {program}")
    if program <= 3 or program == 6:
        return "piano"
    if program <= 7:
        return "electric_piano"
    if program <= 15:
        return "chromatic"
    if program <= 23:
        return "electric_piano"
    if program <= 31:
        return "guitar"
    if program <= 39:
        return "bass"
    if program == 46:
        return "harp"
    if 104 <= program <= 107:
        return "guitar"
    if 108 <= program <= 119:
        return "chromatic"
    # Strings, brass, reeds, pipes, synths, effects, and otherwise unsupported
    # melodic GM programs use the broad synthetic pool. Every brass program
    # (56--63) deliberately maps here.
    return "synth"


def pianoteq_group_programs(group: str) -> frozenset[int]:
    """Return target programs represented by one Pianoteq render component."""

    if group == "drum":
        return frozenset({128})
    return frozenset(
        program for program in range(128) if pianoteq_render_group(program) == group
    )


PIANOTEQ_GROUP_PROGRAMS: dict[str, frozenset[int]] = {
    group: pianoteq_group_programs(group) for group in PIANOTEQ_GROUP_PROGRAM
}
