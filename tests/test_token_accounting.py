import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_system.token_accounting import (
    summarize_token_calls,
    token_call_from_response,
)


class TokenAccountingTest(unittest.TestCase):
    def test_extracts_openai_compatible_usage(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
            )
        )
        call = token_call_from_response(
            response,
            call_type="strategy_synthesis",
            model="strategy-model",
            backend="openai-compatible",
        )

        self.assertEqual(call["input_tokens"], 120)
        self.assertEqual(call["output_tokens"], 30)
        self.assertEqual(call["total_tokens"], 150)
        self.assertTrue(call["usage_available"])

    def test_missing_usage_is_explicit_not_estimated(self):
        call = token_call_from_response(
            SimpleNamespace(),
            call_type="action",
            model="local-model",
            backend="vllm",
        )
        self.assertFalse(call["usage_available"])
        self.assertEqual(call["total_tokens"], 0)

    def test_summary_preserves_call_type_breakdown(self):
        calls = [
            {
                "call_type": "action",
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "usage_available": True,
            },
            {
                "call_type": "strategy_synthesis",
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
                "usage_available": True,
            },
        ]
        summary = summarize_token_calls(calls)

        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["total_tokens"], 36)
        self.assertEqual(summary["by_call_type"]["action"]["total_tokens"], 12)
        self.assertEqual(
            summary["by_call_type"]["strategy_synthesis"]["total_tokens"],
            24,
        )

    def test_policy_and_strategy_backends_record_at_api_boundary(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "prompt_agent"
            / "gpt4o_alfworld.py"
        )
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"get_action_from_gpt", "get_strategy_from_gpt"}
        }

        for method_name, method in methods.items():
            tracker_calls = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "track_api_cost"
            ]
            argument_names = {argument.arg for argument in method.args.kwonlyargs}
            with self.subTest(method_name=method_name):
                self.assertEqual(len(tracker_calls), 2)
                self.assertIn("call_type", argument_names)
                self.assertIn("token_calls", argument_names)

    def test_every_model_call_has_step_attribution(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "prompt_agent"
            / "gpt4o_alfworld.py"
        )
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        model_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get_action_from_gpt", "get_strategy_from_gpt"}
        ]

        self.assertTrue(model_calls)
        for call in model_calls:
            keyword_names = {keyword.arg for keyword in call.keywords}
            with self.subTest(line=call.lineno, method=call.func.attr):
                self.assertIn("call_type", keyword_names)
                self.assertIn("token_calls", keyword_names)

    def test_episode_summary_contains_step_token_calls(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "prompt_agent"
            / "gpt4o_alfworld.py"
        )
        source = script.read_text(encoding="utf-8")
        self.assertIn('"token_calls": instance_token_calls', source)
        self.assertIn('"token_usage": summarize_token_calls(instance_token_calls)', source)
        self.assertIn('timing_stats["token_usage"] = summarize_token_calls', source)


if __name__ == "__main__":
    unittest.main()
