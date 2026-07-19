import ast
import unittest
from pathlib import Path

from index_config import resolve_index_filters, resolve_retrieval_mode


class IndexDefaultTest(unittest.TestCase):
    def test_index_build_defaults_to_successes(self):
        self.assertEqual(resolve_index_filters(), (True, False))
        self.assertEqual(
            resolve_index_filters(correct_only=True),
            (True, False),
        )

    def test_index_ablation_filters_are_explicit(self):
        self.assertEqual(
            resolve_index_filters(failure_only=True),
            (False, True),
        )
        self.assertEqual(
            resolve_index_filters(all_trajectories=True),
            (False, False),
        )
        with self.assertRaises(ValueError):
            resolve_index_filters(correct_only=True, failure_only=True)

    def test_retrieval_defaults_to_success_index(self):
        self.assertEqual(resolve_retrieval_mode(None), "correct_only")
        self.assertEqual(resolve_retrieval_mode(""), "correct_only")
        self.assertEqual(resolve_retrieval_mode("true"), "correct_only")
        self.assertEqual(resolve_retrieval_mode("false"), "all_trajectories")
        self.assertEqual(resolve_retrieval_mode("mix"), "mix")
        with self.assertRaises(ValueError):
            resolve_retrieval_mode("typo")

    def test_all_index_builders_use_shared_default(self):
        repository_root = Path(__file__).resolve().parents[1]
        for script_name in (
            "build_index.py",
            "build_index_traj_level.py",
            "build_index_reasoningbank.py",
        ):
            module = ast.parse(
                (repository_root / script_name).read_text(encoding="utf-8")
            )
            calls_resolver = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "resolve_index_filters"
                for node in ast.walk(module)
            )
            has_all_trajectories_flag = any(
                isinstance(node, ast.Constant)
                and node.value == "--all_trajectories"
                for node in ast.walk(module)
            )
            with self.subTest(script_name=script_name):
                self.assertTrue(calls_resolver)
                self.assertTrue(has_all_trajectories_flag)

    def test_success_index_builders_skip_annotated_duplicates(self):
        repository_root = Path(__file__).resolve().parents[1]
        for script_name in (
            "build_index.py",
            "build_index_traj_level.py",
            "build_index_reasoningbank.py",
        ):
            source = (repository_root / script_name).read_text(encoding="utf-8")
            with self.subTest(script_name=script_name):
                self.assertIn(
                    'correct_only and item.get("is_duplicate_success", False)',
                    source,
                )

    def test_alfworld_runtime_uses_shared_default(self):
        repository_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            Path("utils.py"),
            Path("examples/prompt_agent/gpt4o_alfworld.py"),
        ):
            source = (repository_root / relative_path).read_text(encoding="utf-8")
            module = ast.parse(source)
            calls_resolver = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "resolve_retrieval_mode"
                for node in ast.walk(module)
            )
            with self.subTest(relative_path=str(relative_path)):
                self.assertTrue(calls_resolver)
                self.assertNotIn('CORRECT_ONLY", "false"', source)


if __name__ == "__main__":
    unittest.main()
