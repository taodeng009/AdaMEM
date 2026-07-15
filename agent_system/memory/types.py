"""Canonical public names and backward-compatible aliases for memory modes."""

from typing import Optional


SYNAPSE_MEMORY_TYPES = frozenset(
    {
        "synapse",
        # Legacy public names, including the original misspelling.
        "trajectory_as_examplar_episode",
        "trajectory_as_examplar_episode_correct_only",
        "trajectory_as_exemplar_episode",
        "trajectory_as_exemplar_episode_correct_only",
    }
)


def normalize_memory_type(memory_type: Optional[str]) -> Optional[str]:
    """Return the canonical public memory-mode name."""
    if memory_type in SYNAPSE_MEMORY_TYPES:
        return "synapse"
    return memory_type

