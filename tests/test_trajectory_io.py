import ast
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
            timing = log_dir / "traj_train_newer_timing.json"
            older.write_text("[]", encoding="utf-8")
            newer.write_text("[]", encoding="utf-8")
            timing.write_text('{"total_instances": 1}', encoding="utf-8")
            os.utime(older, ns=(1, 1))
            os.utime(newer, ns=(2, 2))
            os.utime(timing, ns=(3, 3))

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

    def test_alfworld_writer_and_index_builders_share_log_layout(self):
        repository_root = Path(__file__).resolve().parents[1]
        writer_path = (
            repository_root
            / "examples"
            / "prompt_agent"
            / "gpt4o_alfworld.py"
        )
        writer_source = writer_path.read_text(encoding="utf-8")

        self.assertIn('logs/alfworld/{MODEL_NAME.replace(\'/\', \'_\')}', writer_source)
        self.assertNotIn("logs/alfworld_old", writer_source)

        for script_name in (
            "build_index.py",
            "build_index_traj_level.py",
            "build_index_reasoningbank.py",
        ):
            script_path = repository_root / script_name
            script_source = script_path.read_text(encoding="utf-8")
            module = ast.parse(script_source)

            imports_resolver = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "trajectory_io"
                and any(
                    alias.name == "resolve_trajectory_file"
                    for alias in node.names
                )
                for node in module.body
            )
            calls_resolver = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "resolve_trajectory_file"
                for node in ast.walk(module)
            )

            with self.subTest(script_name=script_name):
                self.assertTrue(imports_resolver)
                self.assertTrue(calls_resolver)
                self.assertNotIn("logs/alfworld_old", script_source)


if __name__ == "__main__":
    unittest.main()
