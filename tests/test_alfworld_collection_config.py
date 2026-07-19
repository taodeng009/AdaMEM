"""Regression tests for controllable ALFWorld trajectory collection."""

import ast
import logging
import os
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "prompt_agent"
    / "gpt4o_alfworld.py"
)


def _load_parse_env_int():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_env_int"
    )
    namespace = {"os": os, "logging": logging}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(SCRIPT), "exec"),
        namespace,
    )
    return namespace["_parse_env_int"]


def _load_collection_batch_seed():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_collection_batch_seed"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(SCRIPT), "exec"),
        namespace,
    )
    return namespace["_collection_batch_seed"]


class AlfworldCollectionConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(
            SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT)
        )
        cls.main = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "main"
        )

    def test_training_rounds_are_read_from_test_times(self):
        assignments = [
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "test_times"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)

        value = assignments[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Name)
        self.assertEqual(value.func.id, "_parse_env_int")
        self.assertEqual(value.args[0].value, "TEST_TIMES")

        # The historical train/eval defaults remain fallbacks only; the
        # environment variable is now read for both splits.
        fallback = value.args[1]
        self.assertIsInstance(fallback, ast.IfExp)
        self.assertEqual(fallback.body.value, 1000)
        self.assertEqual(fallback.orelse.value, 3)
        minimum = next(
            keyword.value
            for keyword in value.keywords
            if keyword.arg == "minimum"
        )
        self.assertEqual(minimum.value, 1)

    def test_test_times_accepts_a_small_positive_limit(self):
        parse_env_int = _load_parse_env_int()
        with mock.patch.dict(os.environ, {"TEST_TIMES": "2"}):
            self.assertEqual(parse_env_int("TEST_TIMES", 1000, minimum=1), 2)

    def test_test_times_clamps_zero_to_one(self):
        parse_env_int = _load_parse_env_int()
        with mock.patch.dict(os.environ, {"TEST_TIMES": "0"}):
            with self.assertLogs(level="WARNING") as captured:
                value = parse_env_int("TEST_TIMES", 1000, minimum=1)

        self.assertEqual(value, 1)
        self.assertTrue(
            any("TEST_TIMES=0 is below minimum 1" in line for line in captured.output)
        )

    def test_training_worker_seeds_do_not_overlap_across_batches_or_rounds(self):
        batch_seed = _load_collection_batch_seed()
        worker_seeds = []
        env_num = 8
        batch_size = 4

        for round_idx in range(2):
            for batch_start in range(0, env_num, batch_size):
                seed = batch_seed("train", 1, round_idx, env_num, batch_start)
                worker_seeds.extend(range(seed, seed + batch_size))

        self.assertEqual(worker_seeds, list(range(1, 17)))
        self.assertEqual(len(worker_seeds), len(set(worker_seeds)))

    def test_training_seed_mapping_is_independent_of_batch_size(self):
        batch_seed = _load_collection_batch_seed()

        def seeds_for(batch_size):
            result = []
            for batch_start in range(0, 8, batch_size):
                seed = batch_seed("train", 10, 3, 8, batch_start)
                result.extend(range(seed, seed + batch_size))
            return result

        self.assertEqual(seeds_for(2), seeds_for(4))
        self.assertEqual(seeds_for(4), list(range(34, 42)))

    def test_evaluation_keeps_a_fixed_seed(self):
        batch_seed = _load_collection_batch_seed()
        self.assertEqual(
            batch_seed("eval_in_distribution", 7, 0, 140, 0), 7
        )
        self.assertEqual(
            batch_seed("eval_in_distribution", 7, 2, 140, 100), 7
        )

    def test_environment_builder_receives_computed_batch_seed(self):
        build_env = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_env"
        )
        self.assertIn("seed", [argument.arg for argument in build_env.args.args])

        alfworld_calls = [
            node
            for node in ast.walk(build_env)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_alfworld_envs"
        ]
        self.assertEqual(len(alfworld_calls), 2)
        for call in alfworld_calls:
            seed_keyword = next(
                keyword for keyword in call.keywords if keyword.arg == "seed"
            )
            self.assertIsInstance(seed_keyword.value, ast.Name)
            self.assertEqual(seed_keyword.value.id, "seed")

        main_build_calls = [
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_env"
        ]
        self.assertEqual(len(main_build_calls), 1)
        seed_keyword = next(
            keyword
            for keyword in main_build_calls[0].keywords
            if keyword.arg == "seed"
        )
        self.assertIsInstance(seed_keyword.value, ast.Name)
        self.assertEqual(seed_keyword.value.id, "batch_seed")


if __name__ == "__main__":
    unittest.main()
