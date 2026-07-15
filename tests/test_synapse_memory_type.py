import ast
import unittest
from pathlib import Path

from agent_system.memory.types import normalize_memory_type


class SynapseMemoryTypeTest(unittest.TestCase):
    def test_public_synapse_name_is_canonical(self):
        self.assertEqual(normalize_memory_type("synapse"), "synapse")

    def test_legacy_names_remain_compatible(self):
        for legacy_name in (
            "trajectory_as_examplar_episode",
            "trajectory_as_examplar_episode_correct_only",
            "trajectory_as_exemplar_episode",
            "trajectory_as_exemplar_episode_correct_only",
        ):
            with self.subTest(legacy_name=legacy_name):
                self.assertEqual(normalize_memory_type(legacy_name), "synapse")

    def test_other_memory_types_are_unchanged(self):
        self.assertEqual(normalize_memory_type("adamem-low"), "adamem-low")
        self.assertEqual(normalize_memory_type("reasoningbank"), "reasoningbank")
        self.assertIsNone(normalize_memory_type(None))

    def test_environment_managers_route_synapse_to_retrieval(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "agent_system"
            / "environments"
            / "env_manager.py"
        )
        module = ast.parse(source_path.read_text(encoding="utf-8"))

        for class_name in ("AlfWorldEnvironmentManager", "WebshopEnvironmentManager"):
            manager = next(
                node
                for node in module.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )

            normalizes_constructor_value = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "normalize_memory_type"
                for node in ast.walk(manager)
            )
            has_synapse_branch = any(
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Attribute)
                and isinstance(node.left.value, ast.Name)
                and node.left.value.id == "self"
                and node.left.attr == "memory_type"
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "synapse"
                for node in ast.walk(manager)
            )
            uses_synapse_index = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"get_top_k_memories", "get_top_k_memories_mix"}
                and any(
                    keyword.arg == "memory_type"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "synapse"
                    for keyword in node.keywords
                )
                for node in ast.walk(manager)
            )

            with self.subTest(class_name=class_name):
                self.assertTrue(normalizes_constructor_value)
                self.assertTrue(has_synapse_branch)
                self.assertTrue(uses_synapse_index)


if __name__ == "__main__":
    unittest.main()
