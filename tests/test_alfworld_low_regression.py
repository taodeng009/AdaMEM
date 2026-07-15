"""Source-level regression tests for the ALFWorld AdaMEM-LOW loop."""

import ast
import logging
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "prompt_agent"
    / "gpt4o_alfworld.py"
)


def _load_low_method(*, should_refresh):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "get_action_with_adamem_low"
    )
    namespace = {
        "ALFWORLD_ACTION_INSTR": "direct action",
        "ALFWORLD_TEMPLATE_STRATEGY_REFRESH_DECISION": "strategy={current_strategy}",
        "ALFWORLD_TEMPLATE_STRATEGY_GENERATION": "memories={retrieved_exp}",
        "ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY": "new strategy={strategy}",
        "extract_action_from_response": lambda response: "test action",
        "extract_strategy_from_response": lambda response: "new strategy",
        "parse_action_and_refresh": (
            lambda response: ("tentative action", should_refresh, "test decision")
        ),
        "logging": logging,
        "mix_mode": False,
        "time": time,
        "topk": 1,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SCRIPT), "exec"),
        namespace,
    )
    return namespace["get_action_with_adamem_low"]


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


class AdaMemLowCallCountTest(unittest.IsolatedAsyncioTestCase):
    async def test_initial_strategy_makes_one_action_model_call(self):
        method = _load_low_method(should_refresh=False)
        fake_utils = types.ModuleType("utils")
        fake_utils.get_top_k_memories = lambda *args, **kwargs: []
        fake_utils.get_top_k_memories_mix = lambda *args, **kwargs: ([], [])

        class FakeAgent:
            active_strategies = {}
            action_calls = []
            strategy_calls = []
            strategy_max_tokens = 128

            async def get_action_from_gpt(self, prompt):
                self.action_calls.append(prompt)
                return "action response", 0.25

            async def get_strategy_from_gpt(self, prompt):
                self.strategy_calls.append(prompt)
                return "strategy response", 0.5

            def _truncate_retrieval_for_prompt(self, **kwargs):
                return kwargs["retrieval_text"]

        agent = FakeAgent()
        with mock.patch.dict(sys.modules, {"utils": fake_utils}):
            result = await method(agent, "prompt", None, 0, "observation")

        self.assertEqual(len(agent.action_calls), 1)
        self.assertEqual(len(agent.strategy_calls), 1)
        self.assertTrue(result[1])
        self.assertEqual(result[4]["strategy_time"], 0.5)

    async def test_reused_strategy_makes_one_action_model_call(self):
        method = _load_low_method(should_refresh=False)

        class FakeAgent:
            active_strategies = {0: "current strategy"}
            action_calls = []

            async def get_action_from_gpt(self, prompt):
                self.action_calls.append(prompt)
                return "combined response", 0.25

        agent = FakeAgent()
        result = await method(agent, "prompt", None, 0, "observation")

        self.assertEqual(len(agent.action_calls), 1)
        self.assertIn("current strategy", agent.action_calls[0])
        self.assertFalse(result[1])
        self.assertEqual(result[3]["final_action"], "tentative action")
        self.assertEqual(result[4]["action_time"], 0.25)

    async def test_refreshed_strategy_makes_only_required_calls(self):
        method = _load_low_method(should_refresh=True)
        fake_utils = types.ModuleType("utils")
        fake_utils.get_top_k_memories = lambda *args, **kwargs: []
        fake_utils.get_top_k_memories_mix = lambda *args, **kwargs: ([], [])

        class FakeAgent:
            active_strategies = {0: "current strategy"}
            action_calls = []
            strategy_calls = []
            strategy_max_tokens = 128

            async def get_action_from_gpt(self, prompt):
                self.action_calls.append(prompt)
                return "action response", 0.25

            async def get_strategy_from_gpt(self, prompt):
                self.strategy_calls.append(prompt)
                return "strategy response", 0.5

            def _truncate_retrieval_for_prompt(self, **kwargs):
                return kwargs["retrieval_text"]

        agent = FakeAgent()
        with mock.patch.dict(sys.modules, {"utils": fake_utils}):
            result = await method(agent, "prompt", None, 0, "observation")

        self.assertEqual(len(agent.action_calls), 2)
        self.assertEqual(len(agent.strategy_calls), 1)
        self.assertTrue(result[1])
        self.assertEqual(result[4]["strategy_time"], 0.75)


if __name__ == "__main__":
    unittest.main()
