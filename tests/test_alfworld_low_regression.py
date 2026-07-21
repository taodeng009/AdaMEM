"""Source-level regression tests for the ALFWorld AdaMEM-LOW loop."""

import ast
import logging
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from agent_system.token_accounting import summarize_token_calls


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "prompt_agent"
    / "gpt4o_alfworld.py"
)


def _load_low_method(*, should_refresh):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    timing_helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_timing_call",
            "_summarize_timing_calls",
            "_set_wall_clock_time",
            "_attach_token_usage",
        }
    ]
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
        "summarize_token_calls": summarize_token_calls,
    }
    exec(
        compile(
            ast.Module(body=[*timing_helpers, method], type_ignores=[]),
            str(SCRIPT),
            "exec",
        ),
        namespace,
    )
    return namespace["get_action_with_adamem_low"]


def _load_low_static_method():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    timing_helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_timing_call",
            "_summarize_timing_calls",
            "_set_wall_clock_time",
            "_attach_token_usage",
        }
    ]
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "get_action_with_adamem_low_static"
    )
    namespace = {
        "ALFWORLD_ACTION_INSTR": "direct action",
        "ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY": "static strategy={strategy}",
        "extract_action_from_response": lambda response: "test action",
        "logging": logging,
        "summarize_token_calls": summarize_token_calls,
    }
    exec(
        compile(
            ast.Module(body=[*timing_helpers, method], type_ignores=[]),
            str(SCRIPT),
            "exec",
        ),
        namespace,
    )
    return namespace["get_action_with_adamem_low_static"]


def _is_adamem_low_branch(node: ast.If) -> bool:
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "mem_type"
    ):
        return False
    return any(
        isinstance(value, ast.Constant) and value.value == "adamem-low"
        for comparator in test.comparators
        for value in ast.walk(comparator)
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

    def test_timing_summary_separates_call_time_and_wall_clock(self):
        namespace = {}
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "_timing_call",
                "_summarize_timing_calls",
                "_sample_std",
            }
        ]
        namespace["np"] = __import__("numpy")
        exec(
            compile(ast.Module(body=helpers, type_ignores=[]), str(SCRIPT), "exec"),
            namespace,
        )
        call = namespace["_timing_call"]
        summarize = namespace["_summarize_timing_calls"]
        timing = summarize(
            [
                call("retrieve", "retrieval", 0.1),
                call("strategy", "strategy", 0.4),
                call("action", "action", 0.2),
            ],
            wall_clock_time=0.5,
        )

        self.assertAlmostEqual(timing["model_time"], 0.6)
        self.assertAlmostEqual(timing["total_time"], 0.7)
        self.assertAlmostEqual(timing["wall_clock_time"], 0.5)
        self.assertEqual(len(timing["calls"]), 3)
        self.assertEqual(namespace["_sample_std"]([0.5]), 0.0)

    def test_all_adamem_modes_emit_call_based_timing(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        method_names = {
            "get_action_with_adamem_high",
            "get_action_with_adamem_max",
            "get_action_with_adamem_max_without_trajectory",
            "get_action_with_adamem_max_without_strategy",
            "get_action_with_adamem_low",
            "get_action_with_adamem_low_static",
        }
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name in method_names
        }

        self.assertEqual(set(methods), method_names)
        for method_name, method in methods.items():
            summarizes_calls = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_summarize_timing_calls"
                for node in ast.walk(method)
            )
            with self.subTest(method_name=method_name):
                self.assertTrue(summarizes_calls)

    def test_main_loop_records_concurrent_wall_clock(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("action_batch_start = time.perf_counter()", source)
        self.assertIn("_set_wall_clock_time(", source)
        self.assertIn('"wall_clock_time": instance_wall_clock_time', source)
        self.assertIn('"calls": instance_calls', source)


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

            async def get_action_from_gpt(self, prompt, **kwargs):
                self.action_calls.append(prompt)
                kwargs["token_calls"].append({
                    "call_type": kwargs["call_type"],
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "usage_available": True,
                })
                return "action response", 0.25

            async def get_strategy_from_gpt(self, prompt, **kwargs):
                self.strategy_calls.append(prompt)
                kwargs["token_calls"].append({
                    "call_type": kwargs["call_type"],
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "total_tokens": 24,
                    "usage_available": True,
                })
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
        self.assertEqual(
            [call["name"] for call in result[4]["calls"]],
            ["memory_retrieval", "strategy_synthesis", "strategy_guided_action"],
        )
        self.assertEqual(result[4]["token_usage"]["total_tokens"], 36)

    async def test_reused_strategy_makes_one_action_model_call(self):
        method = _load_low_method(should_refresh=False)

        class FakeAgent:
            active_strategies = {0: "current strategy"}
            action_calls = []

            async def get_action_from_gpt(self, prompt, **kwargs):
                self.action_calls.append(prompt)
                kwargs["token_calls"].append({
                    "call_type": kwargs["call_type"],
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "usage_available": True,
                })
                return "combined response", 0.25

        agent = FakeAgent()
        result = await method(agent, "prompt", None, 0, "observation")

        self.assertEqual(len(agent.action_calls), 1)
        self.assertIn("current strategy", agent.action_calls[0])
        self.assertFalse(result[1])
        self.assertEqual(result[3]["final_action"], "tentative action")
        self.assertEqual(result[4]["action_time"], 0.25)
        self.assertEqual(len(result[4]["calls"]), 1)
        self.assertEqual(result[4]["token_usage"]["total_tokens"], 12)

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

            async def get_action_from_gpt(self, prompt, **kwargs):
                self.action_calls.append(prompt)
                kwargs["token_calls"].append({
                    "call_type": kwargs["call_type"],
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "usage_available": True,
                })
                return "action response", 0.25

            async def get_strategy_from_gpt(self, prompt, **kwargs):
                self.strategy_calls.append(prompt)
                kwargs["token_calls"].append({
                    "call_type": kwargs["call_type"],
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "total_tokens": 24,
                    "usage_available": True,
                })
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
        self.assertAlmostEqual(result[4]["model_time"], 1.0)
        self.assertEqual(len(result[4]["calls"]), 4)
        self.assertEqual(result[4]["token_usage"]["total_tokens"], 48)


class AdaMemLowStaticCallCountTest(unittest.IsolatedAsyncioTestCase):
    async def test_initial_step_delegates_to_existing_low_path(self):
        method = _load_low_static_method()
        sentinel = ("initial response", True, "initial_strategy_generation", {}, {})

        class FakeAgent:
            active_strategies = {}
            delegated_args = None

            async def get_action_with_adamem_low(self, *args):
                self.delegated_args = args
                return sentinel

        agent = FakeAgent()
        result = await method(
            agent,
            "prompt",
            "env-manager",
            7,
            "observation",
            "recent history",
        )

        self.assertIs(result, sentinel)
        self.assertEqual(
            agent.delegated_args,
            ("prompt", "env-manager", 7, "observation", "recent history"),
        )

    async def test_subsequent_steps_reuse_strategy_without_refresh(self):
        method = _load_low_static_method()

        class FakeAgent:
            active_strategies = {0: "initial fixed strategy"}
            action_calls = []

            async def get_action_from_gpt(self, prompt, **kwargs):
                self.action_calls.append((prompt, kwargs["call_type"]))
                kwargs["token_calls"].append(
                    {
                        "call_type": kwargs["call_type"],
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                        "usage_available": True,
                    }
                )
                # A stray refresh tag must not trigger any refresh in static mode.
                return "<action>go north</action><refresh_decision>yes</refresh_decision>", 0.25

        agent = FakeAgent()
        first_result = await method(agent, "prompt", None, 0, "observation")
        result = await method(agent, "next prompt", None, 0, "next observation")

        self.assertEqual(len(agent.action_calls), 2)
        self.assertIn("initial fixed strategy", agent.action_calls[0][0])
        self.assertTrue(
            all(
                call_type == "static_strategy_guided_action"
                for _, call_type in agent.action_calls
            )
        )
        self.assertEqual(agent.active_strategies[0], "initial fixed strategy")
        self.assertFalse(first_result[1])
        self.assertEqual(first_result[2], "static_strategy_reused")
        self.assertFalse(result[1])
        self.assertEqual(result[2], "static_strategy_reused")
        self.assertFalse(result[3]["refresh_decision"])
        self.assertIsNone(result[3]["refresh_prompt"])
        self.assertIsNone(result[3]["strategy_prompt"])
        self.assertEqual(result[3]["strategy"], "initial fixed strategy")
        self.assertEqual(
            [call["name"] for call in result[4]["calls"]],
            ["static_strategy_guided_action"],
        )
        self.assertEqual(result[4]["token_usage"]["total_tokens"], 12)

    def test_static_mode_is_registered_without_replacing_low(self):
        source = SCRIPT.read_text(encoding="utf-8")
        utils_source = (SCRIPT.parents[2] / "utils.py").read_text(encoding="utf-8")

        self.assertIn('"adamem-low"', source)
        self.assertIn('"adamem-low-static"', source)
        self.assertIn("agent.get_action_with_adamem_low_static", source)
        self.assertIn('"adamem-low-static"', utils_source)

if __name__ == "__main__":
    unittest.main()
