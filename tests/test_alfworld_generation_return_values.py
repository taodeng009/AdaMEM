"""Regression tests for the timed model-generation return contract."""

import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "prompt_agent"
    / "gpt4o_alfworld.py"
)

MEMORY_METHODS = {
    "get_action_with_adamem_high",
    "get_action_with_adamem_max",
    "get_action_with_adamem_max_without_trajectory",
    "get_action_with_adamem_max_without_strategy",
}
TIMED_GENERATORS = {"get_action_from_gpt", "get_strategy_from_gpt"}


class TimedGenerationReturnValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(
            SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT)
        )
        cls.parents = {
            child: parent
            for parent in ast.walk(cls.tree)
            for child in ast.iter_child_nodes(parent)
        }
        cls.methods = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name in MEMORY_METHODS
        }

    def test_high_and_max_unpack_text_and_elapsed_time(self):
        self.assertEqual(set(self.methods), MEMORY_METHODS)

        for method_name, method in self.methods.items():
            calls = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in TIMED_GENERATORS
            ]
            self.assertTrue(calls, f"No timed model calls found in {method_name}")

            for call in calls:
                await_node = self.parents.get(call)
                assignment = self.parents.get(await_node)
                self.assertIsInstance(await_node, ast.Await)
                self.assertIsInstance(
                    assignment,
                    ast.Assign,
                    f"{method_name} must unpack the timed model return value",
                )
                target = assignment.targets[0]
                self.assertIsInstance(target, ast.Tuple)
                self.assertEqual(
                    len(target.elts),
                    2,
                    f"{method_name} must unpack (text, elapsed_time)",
                )

    def test_high_and_max_keep_four_item_public_return_contract(self):
        for method_name, method in self.methods.items():
            returns = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
            ]
            self.assertTrue(returns, f"No returns found in {method_name}")
            self.assertTrue(
                all(len(node.value.elts) == 4 for node in returns),
                f"{method_name} must always return four items to the main loop",
            )


if __name__ == "__main__":
    unittest.main()
