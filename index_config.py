"""Shared defaults for building and loading offline memory indexes."""

from typing import Optional, Tuple


def resolve_index_filters(
    *,
    correct_only: bool = False,
    failure_only: bool = False,
    all_trajectories: bool = False,
) -> Tuple[bool, bool]:
    """Return success/failure filters, defaulting to successful trajectories."""
    if sum((correct_only, failure_only, all_trajectories)) > 1:
        raise ValueError(
            "--correct_only, --failure_only and --all_trajectories are "
            "mutually exclusive"
        )
    if failure_only:
        return False, True
    if all_trajectories:
        return False, False
    return True, False


def resolve_retrieval_mode(value: Optional[str]) -> str:
    """Normalize CORRECT_ONLY, defaulting to the successful-trajectory index."""
    normalized = (value or "true").strip().lower()
    aliases = {
        "true": "correct_only",
        "correct_only": "correct_only",
        "success": "correct_only",
        "false": "all_trajectories",
        "all": "all_trajectories",
        "all_trajectories": "all_trajectories",
        "mix": "mix",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "CORRECT_ONLY must be true, false, or mix; "
            f"got {value!r}"
        ) from exc
