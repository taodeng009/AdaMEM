import os
import tempfile
import unittest
from pathlib import Path

from trajectory_io import resolve_trajectory_file


class ResolveTrajectoryFileTest(unittest.TestCase):
    def test_defaults_to_new_log_directory_and_selects_newest_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs" / "alfworld" / "Qwen_Model"
            log_dir.mkdir(parents=True)
            older = log_dir / "traj_train_older.json"
            newer = log_dir / "traj_train_newer.json"
            older.write_text("[]", encoding="utf-8")
            newer.write_text("[]", encoding="utf-8")
            os.utime(older, ns=(1, 1))
            os.utime(newer, ns=(2, 2))

            selected = resolve_trajectory_file(
                "alfworld", "Qwen/Model", root=root
            )

            self.assertEqual(selected, newer)

    def test_explicit_trajectory_file_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "custom.json"
            explicit.write_text("[]", encoding="utf-8")

            selected = resolve_trajectory_file(
                "alfworld",
                "Qwen/Model",
                trajectory_file="custom.json",
                root=root,
            )

            self.assertEqual(selected, explicit)


if __name__ == "__main__":
    unittest.main()
