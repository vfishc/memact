import copy
import unittest
import tempfile
import json
import struct
from pathlib import Path

from PIL import Image
from evaluate_routed_memory_qa import FrameResolver, score_answers, score_memory

from memory_runtime import MemoryManager, digest
from routed_facts import EpisodeFacts
from routed_qa import CURRENT, MEMORY, NoSample, QAGenerator
from routed_stream import build_stream, check_public, evaluation_schedule, run_session, validate_stream
from qa_semantic_contract import clean_category_options, visual_category
from routed_intrinsics import sim_intrinsics, resize_intrinsics, scannet_intrinsics, scannetpp_intrinsics


def fixture_record():
    objects = {
        "lamp.1": {"category": "cabinet", "world_center": [-1, 0, -2]},
        "chair.1": {"category": "chair", "world_center": [0, 0, -3]},
        "chair.2": {"category": "chair", "world_center": [2, 0, -3]},
        "table.1": {"category": "table", "world_center": [-1, 0, -5]},
        "sofa.1": {"category": "sofa", "world_center": [4, 0, -6]},
    }
    visibility = []
    for index in range(22):
        visible = ["lamp.1"]
        if index >= 2:
            visible += ["chair.1", "table.1"]
        if index >= 4:
            visible += ["sofa.1"]
        if 6 <= index < 10 or 15 <= index:
            visible += ["chair.2"]
        if 10 <= index < 14 or index >= 18:
            visible.remove("lamp.1")
        visibility.append({"frame_id": index, "visible_objects": visible})
    return {"source": "sim_front3d", "fps": 1, "episode_id": "fixture", "scene_id": "scene", "split": "train",
            "objects": objects, "resolution": [640, 480], "K": [[300, 0, 320], [0, 300, 240], [0, 0, 1]],
            "camera": {"pose_lookup": [{"location": [0.05 * index, 0, 0], "rotation_euler_deg": [0, 0, 0]} for index in range(22)]},
            "visibility": {"frames": visibility}}


def filled_manager():
    manager = MemoryManager(page_size=2, frame_budget=1)
    for index in range(5):
        manager.observe(f"f{index}", index)
    payload = {"type": "memory_write", "observations": [{"frame": f"f{index}", "objects": [{"category": "chair", "count": 1}]} for index in range(5)], "relations": []}
    manager.write(payload, 0, 5)
    return manager, payload


class MemoryTests(unittest.TestCase):
    def test_middle_priority_then_bounded_pagination(self):
        manager, _ = filled_manager()
        request = {"type": "use_memory", "categories": ["chair"], "with_frames": True}
        result = manager.query(request)
        self.assertEqual(result["frames"][0]["frame"], "f2")
        returned = []
        while True:
            returned += [item["frame"] for item in result["frames"]]
            if not result["next_cursor"]:
                break
            request["cursor"] = result["next_cursor"]
            result = manager.query(request)
        self.assertEqual(len(set(returned)), 4)
        self.assertEqual(len(returned), 4)
        self.assertEqual(result["remaining_frame_budget"], 0)

    def test_new_queries_and_cursor_retries_cannot_bypass_budget(self):
        manager, _ = filled_manager()
        request = {"type": "use_memory", "categories": ["chair"], "with_frames": True}
        first = manager.query(request)
        returned = [row["frame"] for row in first["frames"]]
        for index in range(10):
            query = request if index % 2 else {**request, "cursor": first["next_cursor"]}
            result = manager.query(query)
            self.assertLessEqual(len(result["frames"]), 1)
            returned.extend(row["frame"] for row in result["frames"])
        self.assertEqual(len(returned), 4)
        self.assertEqual(len(set(returned)), 4)
        manager.begin_question()
        self.assertEqual(len(manager.query(request)["frames"]), 1)
        with self.assertRaises(ValueError):
            manager.query({**request, "cursor": first["next_cursor"]})

    def test_text_only_and_no_gt_repair(self):
        manager, payload = filled_manager()
        payload["observations"][0]["objects"][0]["category"] = "sofa"
        other = MemoryManager()
        for frame, timestamp in manager.frames.items():
            other.observe(frame, timestamp)
        other.write(payload, 0, 5)
        result = other.query({"type": "use_memory", "categories": ["sofa"], "with_frames": False})
        self.assertEqual(result["records"][0]["objects"][0]["category"], "sofa")
        self.assertEqual(result["frames"], [])

    def test_idempotence_conflict_and_missing_frame(self):
        manager, payload = filled_manager()
        self.assertEqual(manager.write(payload, 0, 5)["status"], "already_stored")
        payload["observations"][0]["objects"][0]["count"] = 2
        with self.assertRaises(ValueError):
            manager.write(payload, 0, 5)
        fresh = MemoryManager()
        fresh.observe("f0", 0)
        with self.assertRaises(ValueError):
            fresh.write({"type": "memory_write", "observations": [], "relations": []}, 0, 5)
        with self.assertRaises(ValueError):
            fresh.write({"type": "memory_write", "observations": ["invalid"], "relations": []}, 0, 5)

    def test_cursor_snapshot_and_query_binding(self):
        manager, _ = filled_manager()
        request = {"type": "use_memory", "categories": ["chair"], "with_frames": False}
        cursor = manager.query(request)["next_cursor"]
        manager.observe("f5", 5)
        manager.write({"type": "memory_write", "observations": [{"frame": "f5", "objects": [{"category": "chair", "count": 9}]}], "relations": []}, 5, 6)
        result = manager.query({**request, "cursor": cursor})
        self.assertEqual(result["snapshot_windows"], 1)
        self.assertEqual(result["total_records"], 5)
        with self.assertRaises(ValueError):
            manager.query({**request, "with_frames": True, "cursor": cursor})

    def test_future_unknown_fields_and_replay(self):
        manager, _ = filled_manager()
        with self.assertRaises(ValueError):
            manager.query({"type": "use_memory", "categories": [], "with_frames": True, "path": "/secret"})
        result = manager.query({"type": "use_memory", "categories": [], "with_frames": True, "time_range": [6, 9]})
        self.assertEqual(result["records"], [])
        self.assertEqual(manager.replay().summaries, manager.summaries)


class QATests(unittest.TestCase):
    def setUp(self):
        self.facts = EpisodeFacts(fixture_record(), 1, 22)
        self.generator = QAGenerator(self.facts)

    def test_distinct_count_not_observation_sum(self):
        sample = self.generator.make("seen_count_by_category", 11)
        self.assertEqual(sample["turns"][0]["answer"], 2)
        self.assertTrue(sample["turns"][0]["with_frames"])

    def test_full_bank_retrieval_bounds_and_replay(self):
        events, audit, rejected = build_stream(self.facts, self.generator.generate())
        validate_stream(events, {row["frame"]: {} for row in self.facts.frames})
        images = []
        for event in events:
            if event["kind"] == "question":
                images = []
            if event["kind"] == "memory_result":
                frames = [row["frame"] for row in event["payload"]["frames"]]
                self.assertLessEqual(len(frames), 2)
                images.extend(frames)
                self.assertLessEqual(len(images), 4)
                self.assertEqual(len(images), len(set(images)))
        for row in audit:
            results = [event["payload"] for event in events[row["question_seq"]:row["answer_seq"]] if event["kind"] == "memory_result"]
            if row["template"].startswith("single_seen_object_then"):
                self.assertEqual(sum(len(result["frames"]) for result in results), 1)
            if row["template"] in ("first_seen_object_category", "seen_category_order", "first_seen_timestamp_s", "first_seen_category_then_timestamp"):
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["frames"], [])

    def test_first_time_preserved_not_midpoint(self):
        sample = self.generator.make("first_seen_timestamp_s", 11)
        self.assertEqual(sample["turns"][0]["answer"], 0)
        self.assertFalse(sample["turns"][0]["with_frames"])

    def test_single_seen_repeated_visit_allowed(self):
        sample = self.generator.make("single_seen_object_then_closest_category", 20)
        self.assertEqual(sample["targets"][0], "lamp.1")
        self.assertEqual(sample["turns"][0]["answer"], "chair")
        distance = self.generator.make("single_seen_object_then_closest_distance", 20)
        self.assertAlmostEqual(distance["turns"][0]["answer"], 1.414)

    def test_multiple_historical_category_instances_rejected(self):
        record = fixture_record()
        record["objects"]["lamp.2"] = {"category": "cabinet", "world_center": [5, 0, -8]}
        record["visibility"]["frames"][3]["visible_objects"].append("lamp.2")
        generator = QAGenerator(EpisodeFacts(record, 1, 22))
        with self.assertRaises(NoSample):
            generator.make("single_seen_object_then_closest_category", 20)

    def test_future_instance_does_not_change_prefix(self):
        record = fixture_record()
        record["objects"]["lamp.2"] = {"category": "cabinet", "world_center": [5, 0, -8]}
        record["visibility"]["frames"][21]["visible_objects"].append("lamp.2")
        generator = QAGenerator(EpisodeFacts(record, 1, 22))
        self.assertEqual(generator.make("single_seen_object_then_closest_category", 20), self.generator.make("single_seen_object_then_closest_category", 20))

    def test_teacher_prefix_causal_and_regions_optional(self):
        teacher = self.facts.teacher(range(5))
        record = fixture_record()
        record["visibility"]["frames"][21]["visible_objects"] = []
        self.assertEqual(teacher, EpisodeFacts(record, 1, 22).teacher(range(5)))
        self.assertTrue(teacher["relations"])
        check_public(teacher)

    def test_two_turn_chains_and_later_time(self):
        samples = self.generator.generate()
        for sample in samples:
            if "then" in sample["template"] and sample["template"] in CURRENT + ("first_seen_category_then_timestamp",):
                self.assertEqual(len(sample["turns"]), 2)
        later = [sample for sample in samples if sample["template"] == "object_category_then_later_visibility"]
        self.assertTrue(later)
        self.assertTrue(all(sample["turns"][1]["index"] >= sample["turns"][0]["index"] + 5 for sample in later))

    def test_only_clean_four_or_five_category_options(self):
        for sample in self.generator.generate():
            for turn in sample["turns"]:
                if turn["options"] is not None:
                    self.assertIn(len(turn["options"]), (4, 5))
                    self.assertIn(turn["answer"], turn["options"])
                    self.assertTrue(all(visual_category(option) == option for option in turn["options"]))
                    _, cleaned, _ = clean_category_options(turn["answer"], turn["options"])
                    self.assertEqual(set(cleaned), set(turn["options"]))
        self.assertIsNone(self.generator.make("camera_path_length_m", 10)["turns"][0]["options"])

    def test_old_category_filter_and_family_conflicts(self):
        answer, options, reasons = clean_category_options("chair", ["chair", "armchair", "object", "wall", "table", "bed", "sofa", "chair"])
        self.assertEqual(answer, "chair")
        self.assertNotIn("object", options)
        self.assertNotIn("wall", options)
        self.assertTrue(reasons)

    def test_all_twenty_one_templates_exercised(self):
        record = fixture_record()
        for index, pose in enumerate(record["camera"]["pose_lookup"]):
            pose["location"] = [0.4 * index, 0, 0]
        alternative = QAGenerator(EpisodeFacts(record, 1, 22))
        templates = {sample["template"] for sample in self.generator.generate() + alternative.generate()}
        self.assertEqual(templates, set(CURRENT + MEMORY))

    def test_tied_global_first_allowed_but_options_unique(self):
        record = fixture_record()
        record["objects"]["lamp.1"]["world_center"] = [0.8, 0, -2]
        record["visibility"]["frames"][0]["visible_objects"].append("chair.1")
        generator = QAGenerator(EpisodeFacts(record, 1, 22))
        for template in ("first_seen_object_category", "seen_category_order", "first_seen_category_then_timestamp"):
            sample = generator.make(template, 16)
            turn = sample["turns"][0]
            self.assertEqual(turn["answer"], "cabinet")
            self.assertNotIn("chair", turn["options"])
            self.assertIn("listed", turn["question"])

    def test_absent_distractors_and_semantic_duplicate_marker(self):
        first = self.generator.make("first_seen_object_category", 11)
        order = self.generator.make("seen_category_order", 11)
        self.assertEqual(first["semantic_key"], order["semantic_key"])
        choices = first["turns"][0]["options"]
        scene_categories = set(self.facts.categories.values())
        self.assertTrue(set(choices) - scene_categories)
        self.assertIn(first["turns"][0]["answer"], scene_categories)
        for choice in set(choices) - scene_categories:
            self.assertIn(choice, self.generator.absent_categories)

    def test_first_timestamp_does_not_require_current_visibility(self):
        sample = self.generator.make("first_seen_category_then_timestamp", 20)
        self.assertFalse(set(sample["targets"]) & self.facts.frames[20]["visible"])
        self.assertEqual(len(sample["turns"]), 2)
        self.assertEqual(sample["turns"][0]["answer"], "cabinet")
        self.assertEqual(sample["turns"][1]["answer"], 0)
        self.assertTrue(all(turn["memory"] and not turn["with_frames"] for turn in sample["turns"]))

    def test_first_timestamp_allows_multiple_instances_and_nonzero_time(self):
        record = fixture_record()
        record["objects"]["lamp.2"] = {"category": "cabinet", "world_center": [1, 0, -2]}
        for frame in record["visibility"]["frames"]:
            if frame["frame_id"] < 3:
                frame["visible_objects"] = []
            elif "lamp.1" in frame["visible_objects"]:
                frame["visible_objects"].append("lamp.2")
        generator = QAGenerator(EpisodeFacts(record, 1, 22))
        sample = generator.make("first_seen_category_then_timestamp", 20)
        self.assertEqual(sample["turns"][0]["answer"], "cabinet")
        self.assertEqual(sample["turns"][1]["answer"], 3)
        self.assertEqual(len(sample["targets"]), 2)

    def test_missing_boolean_frame_fails_closed(self):
        record = fixture_record()
        record["visibility"]["frames"].pop(5)
        with self.assertRaises(ValueError):
            EpisodeFacts(record)

    def test_motion_numerical_truth(self):
        sample = self.generator.make("camera_path_length_m", 10)
        self.assertAlmostEqual(sample["turns"][0]["answer"], 0.25)
        self.assertEqual(self.generator.make("camera_rotation_deg", 10)["turns"][0]["answer"], 0)


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.facts = EpisodeFacts(fixture_record(), 1, 17)
        self.samples = QAGenerator(self.facts).generate()
        self.events, self.audit, _ = build_stream(self.facts, self.samples)
        self.media = {row["frame"]: {} for row in self.facts.frames}

    def test_seventeen_seconds_four_writes(self):
        check = validate_stream(self.events, self.media)
        self.assertEqual(check["memory_writes"], 4)
        self.assertEqual(check["observations"], 17)
        self.assertGreater(check["answers"], 0)
        windows = [event["payload"]["window"] for event in self.events if event["kind"] == "write_prompt"]
        self.assertEqual(windows, [[0, 5], [5, 10], [10, 15], [15, 17]])

    def test_gt_leak_and_fake_tool_result_detected(self):
        changed = copy.deepcopy(self.events)
        changed[0]["payload"]["pose"] = [0, 0, 0]
        with self.assertRaises(ValueError):
            validate_stream(changed, self.media)
        changed = copy.deepcopy(self.events)
        position = next(index for index, event in enumerate(changed) if event["kind"] == "memory_result")
        changed[position]["payload"]["total_records"] += 1
        for index in range(1, len(changed)):
            changed[index]["prefix_hash"] = digest(changed[index - 1])
        with self.assertRaises(ValueError):
            validate_stream(changed, self.media)

    def test_fake_model_shared_context_without_teacher_schedule(self):
        schedule, expected = evaluation_schedule(self.events, self.audit)
        outputs = iter([event["payload"] for event in self.events if event["role"] == "assistant"])
        lengths = []
        def fake_model(context):
            lengths.append(len(context))
            return copy.deepcopy(next(outputs))
        predictions, context, manager = run_session(schedule, fake_model)
        self.assertEqual([row["answer"] for row in predictions], [row["answer"] for row in expected])
        self.assertTrue(all(after > before for before, after in zip(lengths, lengths[1:])))
        self.assertEqual(len(manager.summaries), 4)
        self.assertTrue(context)
        self.assertTrue(all("answer" not in event["payload"] for event in schedule))

    def test_free_running_wrong_memory_is_not_repaired(self):
        schedule = [{"kind": "observation", "payload": {"frame": "f0", "timestamp": 0}},
                    {"kind": "write_prompt", "payload": {"mode": "[MEMORY_WRITE]", "window": [0, 5], "frames": ["f0"]}},
                    {"kind": "question", "payload": {"mode": "[QA]", "question": "What was seen?"}}]
        outputs = iter([{"type": "memory_write", "observations": [{"frame": "f0", "objects": [{"category": "wrong category", "count": 7}]}], "relations": []},
                        {"type": "use_memory", "categories": ["chair"], "with_frames": True},
                        {"type": "answer", "answer": "unknown"}])
        predictions, context, manager = run_session(schedule, lambda context: next(outputs))
        result = next(message["content"] for message in context if message["role"] == "tool" and "records" in message["content"])
        self.assertEqual(result["records"], [])
        self.assertEqual(predictions[0]["answer"], "unknown")
        self.assertEqual(manager.summaries[0]["payload"]["observations"][0]["objects"][0]["count"], 7)

    def test_schema_error_returns_to_model_for_repair(self):
        schedule = [{"kind": "observation", "payload": {"frame": "f0", "timestamp": 0}},
                    {"kind": "write_prompt", "payload": {"mode": "[MEMORY_WRITE]", "window": [0, 5], "frames": ["f0"]}}]
        outputs = iter([{"type": "memory_write", "observations": [], "relations": []},
                        {"type": "memory_write", "observations": [{"frame": "f0", "objects": []}], "relations": []}])
        _, context, manager = run_session(schedule, lambda context: next(outputs))
        self.assertEqual(len(manager.summaries), 1)
        self.assertTrue(any(message["content"].get("status") == "invalid_memory" for message in context))


class EvaluationTests(unittest.TestCase):
    def test_score_missing_and_wrong_numeric(self):
        expected = [{"schedule_index": 1, "template": "distance", "sample": 0, "answer": 2.0},
                    {"schedule_index": 2, "template": "category", "sample": 0, "answer": "chair"}]
        report = score_answers(expected, [{"schedule_index": 1, "answer": 3.0}])
        self.assertEqual(report["templates"]["distance"]["mean_absolute_error"], 1)
        self.assertEqual(report["templates"]["category"]["missing"], 1)
        self.assertEqual(report["two_turn_exact_success"], 0)

    def test_memory_count_metric(self):
        _, truth = filled_manager()
        prediction = copy.deepcopy(truth)
        prediction["observations"][0]["objects"][0]["count"] = 3
        report = score_memory([truth], [prediction])
        self.assertEqual(report["category_recall"], 1)
        self.assertEqual(report["frame_category_count_mae"], 0.4)

    def test_frame_resolver_returns_rgb_without_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (8, 6), "red").save(path)
            resolver = FrameResolver({"f0": {"path": str(path)}})
            messages = [{"role": "user", "content": {"frame": "f0", "timestamp": 0}},
                        {"role": "tool", "content": {"frames": [{"frame": "f0", "timestamp": 0}]}}]
            actual = resolver.materialize(messages)
            self.assertEqual(actual[1]["images"][0]["image"].getpixel((0, 0)), (255, 0, 0))
            with self.assertRaises(ValueError):
                resolver.materialize(messages[1:])


class IntrinsicsTests(unittest.TestCase):
    def test_sim_settings_require_native_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "settings.py"
            settings.write_text("RES_X = 768\nRES_Y = 768\nFOCAL_LENGTH = 24\n")
            matrix, provenance = sim_intrinsics(settings, [768, 768])
            self.assertEqual(matrix, [[512, 0, 384], [0, 512, 384], [0, 0, 1]])
            self.assertFalse(provenance["per_episode_job_receipt"])
            with self.assertRaises(ValueError):
                sim_intrinsics(settings, [640, 480])

    def test_anisotropic_resize(self):
        self.assertEqual(resize_intrinsics([[100, 0, 50], [0, 200, 100], [0, 0, 1]], [100, 200], [200, 100]),
                         [[200, 0, 100], [0, 100, 50], [0, 0, 1]])

    def test_scannet_sensor_header(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.sens"
            matrix = np.eye(4, dtype="<f4")
            matrix[0, 0] = matrix[1, 1] = 1000
            matrix[0, 2], matrix[1, 2] = 640, 480
            path.write_bytes(struct.pack("<IQ", 4, 3) + b"cam" + matrix.tobytes() + bytes(64 * 3) + struct.pack("<iiIIIIf", 2, 1, 1280, 960, 640, 480, 1000))
            self.assertEqual(scannet_intrinsics(path, [640, 480]), [[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    def test_scannetpp_uses_source_time_not_chunk_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose_intrinsic_imu.json"
            path.write_text(json.dumps({"90.0": {"intrinsic": [1000, 0, 640, 0, 1000, 480, 0, 0, 1]}}))
            self.assertEqual(scannetpp_intrinsics(path, [90.01], [1280, 960], [640, 480])[0][0][0], 500)
            with self.assertRaises(ValueError):
                scannetpp_intrinsics(path, [0], [1280, 960], [640, 480])


if __name__ == "__main__":
    unittest.main()
