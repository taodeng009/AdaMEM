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


if __name__ == "__main__":
    unittest.main()
