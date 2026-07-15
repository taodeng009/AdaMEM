"""Source-level regression tests for the ALFWorld AdaMEM-LOW loop."""

import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "prompt_agent"
    / "gpt4o_alfworld.py"
)


def _is_adamem_low_branch(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "mem_type"
        and any(
            isinstance(comparator, ast.Constant)
            and comparator.value == "adamem-low"
            for comparator in test.comparators
        )
    )


class AdaMemLowRegressionTest(unittest.TestCase):
    def test_adamem_low_reads_current_batch_trajectories(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and _is_adamem_low_branch(node)
        ]
        self.assertTrue(branches, "AdaMEM-LOW branch was not found")

        loaded_names = {
            node.id
            for branch in branches
            for node in ast.walk(branch)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertIn("batch_trajs", loaded_names)
        self.assertNotIn("trajs", loaded_names)


if __name__ == "__main__":
    unittest.main()
