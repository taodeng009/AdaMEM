"""Utilities shared by the offline memory-index builders."""

from glob import glob
from pathlib import Path
from typing import Optional, Union


def resolve_trajectory_file(
    dataset_name: str,
    base_model_name: str,
    trajectory_file: Optional[str] = None,
    root: Union[str, Path] = ".",
) -> Path:
    """Resolve an explicit trajectory path or the newest training rollout log."""
    root = Path(root)
    base_model_safe = base_model_name.replace("/", "_")

    if trajectory_file:
        pattern = Path(trajectory_file)
        if not pattern.is_absolute():
            pattern = root / pattern
    else:
        pattern = (
            root
            / "logs"
            / dataset_name
            / base_model_safe
            / "traj_train*.json"
        )

    candidates = [Path(path) for path in glob(str(pattern)) if Path(path).is_file()]
    if trajectory_file is None:
        # Rollouts also emit traj_train*_timing.json files. They match the
        # default glob but contain timing statistics rather than trajectories.
        candidates = [
            path for path in candidates if not path.name.endswith("_timing.json")
        ]
    if not candidates:
        raise FileNotFoundError(
            "No training trajectory file found. Looked for: "
            f"{pattern}. Run a training-split rollout first or pass --traj_file."
        )

    selected = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    if len(candidates) > 1:
        print(
            f"Found {len(candidates)} matching trajectory files; "
            f"using newest: {selected}"
        )
    return selected
