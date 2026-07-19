"""Regression tests for controllable ALFWorld trajectory collection."""

import ast
import hashlib
import json
import logging
import os
import re
import tempfile
import unittest
import uuid
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


def _load_trajectory_collection_helpers():
    names = {
        "_new_trajectory_record",
        "_normalize_gamefile_for_identity",
        "_normalize_action_for_identity",
        "_trajectory_identity_hash",
        "_record_unique_successes",
        "_success_target_reached",
        "_load_trajectory_checkpoint",
        "_atomic_write_trajectory_checkpoint",
        "_restore_collection_state",
    }
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "re": re,
        "uuid": uuid,
    }
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(SCRIPT), "exec"),
        namespace,
    )
    return namespace


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

    def test_new_trajectory_records_traceable_episode_metadata(self):
        helpers = _load_trajectory_collection_helpers()
        record = helpers["_new_trajectory_record"](
            episode_id=17,
            round_idx=2,
            seed=101,
            gamefile="/data/alfworld/game.tw-pddl",
        )

        self.assertEqual(record["episode_id"], 17)
        self.assertEqual(record["round_idx"], 2)
        self.assertEqual(record["seed"], 101)
        self.assertEqual(record["gamefile"], "/data/alfworld/game.tw-pddl")
        self.assertIsNone(record["won"])
        self.assertEqual(record["steps"], [])

    def test_duplicate_successes_are_annotated_and_not_counted_twice(self):
        helpers = _load_trajectory_collection_helpers()
        make_record = helpers["_new_trajectory_record"]
        record_successes = helpers["_record_unique_successes"]

        def successful_record(episode_id, action):
            record = make_record(
                episode_id=episode_id,
                round_idx=0,
                seed=episode_id + 1,
                gamefile="/data/alfworld/same-game/game.tw-pddl",
            )
            record["won"] = True
            record["steps"] = [{"curr_action": action}]
            return record

        first = successful_record(0, "<action>Open Fridge 1</action>")
        duplicate = successful_record(1, "<action> open   fridge 1 </action>")
        different = successful_record(2, "<action>go to fridge 1</action>")
        failed = successful_record(3, "<action>open fridge 1</action>")
        failed["won"] = False
        seen = set()

        accepted = record_successes(
            [first, duplicate, different, failed], seen
        )

        self.assertEqual(accepted, 2)
        self.assertEqual(len(seen), 2)
        self.assertFalse(first["is_duplicate_success"])
        self.assertTrue(duplicate["is_duplicate_success"])
        self.assertFalse(different["is_duplicate_success"])
        self.assertNotIn("trajectory_hash", failed)

    def test_missing_gamefile_does_not_merge_unrelated_episodes(self):
        helpers = _load_trajectory_collection_helpers()
        make_record = helpers["_new_trajectory_record"]
        record_successes = helpers["_record_unique_successes"]
        records = []
        for episode_id in (10, 11):
            record = make_record(
                episode_id=episode_id,
                round_idx=0,
                seed=episode_id,
                gamefile="",
            )
            record["won"] = True
            record["steps"] = [{"curr_action": "<action>look</action>"}]
            records.append(record)

        self.assertEqual(record_successes(records, set()), 2)

    def test_success_target_is_optional_and_inclusive(self):
        reached = _load_trajectory_collection_helpers()[
            "_success_target_reached"
        ]
        self.assertFalse(reached(0, 500))
        self.assertFalse(reached(200, 199))
        self.assertTrue(reached(200, 200))
        self.assertTrue(reached(200, 203))

    def test_main_loop_uses_metadata_deduplication_and_target_stop(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"TARGET_SUCCESS_TRAJECTORIES"', source)
        self.assertIn("_new_trajectory_record(", source)
        self.assertIn("_record_unique_successes(", source)
        self.assertIn("_success_target_reached(", source)
        self.assertIn("if stop_collection:\n            break", source)

    def test_atomic_checkpoint_round_trip(self):
        helpers = _load_trajectory_collection_helpers()
        write_checkpoint = helpers["_atomic_write_trajectory_checkpoint"]
        load_checkpoint = helpers["_load_trajectory_checkpoint"]
        make_record = helpers["_new_trajectory_record"]
        record = make_record(
            episode_id=0,
            round_idx=0,
            seed=100,
            gamefile="/data/train/game.tw-pddl",
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "traj_train.json"
            write_checkpoint(str(checkpoint), [record])
            restored = load_checkpoint(str(checkpoint))

            self.assertEqual(restored, [record])
            self.assertEqual(list(Path(tmp).glob("*.tmp-*")), [])

    def test_atomic_checkpoint_preserves_previous_file_on_replace_failure(self):
        helpers = _load_trajectory_collection_helpers()
        write_checkpoint = helpers["_atomic_write_trajectory_checkpoint"]

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "traj_train.json"
            checkpoint.write_text('[{"existing": true}]', encoding="utf-8")
            with mock.patch.object(
                os, "replace", side_effect=OSError("simulated interruption")
            ):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    write_checkpoint(str(checkpoint), [{"replacement": True}])

            self.assertEqual(
                json.loads(checkpoint.read_text(encoding="utf-8")),
                [{"existing": True}],
            )
            self.assertEqual(list(Path(tmp).glob("*.tmp-*")), [])

    def test_resume_rejects_legacy_trajectory_without_required_metadata(self):
        load_checkpoint = _load_trajectory_collection_helpers()[
            "_load_trajectory_checkpoint"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.json"
            checkpoint.write_text(
                json.dumps([{"won": True, "steps": []}]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_checkpoint(str(checkpoint))

    def test_resume_restores_deduplication_ids_rounds_and_seeds(self):
        helpers = _load_trajectory_collection_helpers()
        make_record = helpers["_new_trajectory_record"]
        restore_state = helpers["_restore_collection_state"]

        records = []
        for episode_id, round_idx, seed, won, action in (
            (0, 0, 200, True, "<action>look</action>"),
            (1, 0, 201, True, "<action> LOOK </action>"),
            (2, 1, 202, False, "<action>open fridge 1</action>"),
        ):
            record = make_record(
                episode_id=episode_id,
                round_idx=round_idx,
                seed=seed,
                gamefile="/data/train/same-game/game.tw-pddl",
            )
            record["won"] = won
            record["steps"] = [{"curr_action": action}]
            records.append(record)

        state = restore_state(records)

        self.assertEqual(state["unique_success_count"], 1)
        self.assertEqual(len(state["seen_success_hashes"]), 1)
        self.assertFalse(records[0]["is_duplicate_success"])
        self.assertTrue(records[1]["is_duplicate_success"])
        self.assertEqual(state["next_episode_id"], 3)
        self.assertEqual(state["next_round_idx"], 2)
        self.assertEqual(state["next_seed"], 203)

    def test_main_loop_uses_resume_offsets_and_atomic_checkpoints(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"RESUME_TRAJECTORY_FILE"', source)
        self.assertIn("episode_id_offset + test_idx * env_num", source)
        self.assertIn("round_idx=global_round_idx", source)
        self.assertIn("collection_seed_start", source)
        self.assertIn("_load_trajectory_checkpoint(traj_file)", source)
        self.assertIn("_restore_collection_state(traj_items)", source)
        self.assertIn("_atomic_write_trajectory_checkpoint(", source)


if __name__ == "__main__":
    unittest.main()
