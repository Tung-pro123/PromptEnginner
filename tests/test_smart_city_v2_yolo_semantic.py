# -*- coding: utf-8 -*-
"""Tests for the local YOLO sign/light adapter and async worker."""

from __future__ import absolute_import

import os
import sys
import time
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.smart_city.v2.semantic import SemanticObservation  # noqa: E402
from src.smart_city.v2.yolo_semantic import (  # noqa: E402
    LatestFrameSemanticWorker,
    YoloSemanticDetector,
    forbidden_bbox_to_label,
    resolve_semantic_detections,
)


MODEL_NAMES = {
    0: "Corner",
    1: "Decision",
    2: "Forbidden",
    3: "Green_Light",
    4: "Interact",
    5: "Left",
    6: "Red_Light",
    7: "Right",
    8: "straight",
}


def detection(class_name, confidence, centre_x, width=1000):
    del width
    return {
        "class_name": class_name,
        "confidence": confidence,
        "xyxy": [centre_x - 10, 20, centre_x + 10, 60],
    }


class ForbiddenPositionTests(unittest.TestCase):
    def test_left_centre_and_right_zones(self):
        self.assertEqual(
            "NO_LEFT", forbidden_bbox_to_label([100, 0, 200, 50], 1000)
        )
        self.assertEqual(
            "NO_STRAIGHT", forbidden_bbox_to_label([450, 0, 550, 50], 1000)
        )
        self.assertEqual(
            "NO_RIGHT", forbidden_bbox_to_label([800, 0, 900, 50], 1000)
        )

    def test_dead_band_and_invalid_box_fail_closed(self):
        self.assertIsNone(
            forbidden_bbox_to_label([390, 0, 410, 50], 1000)
        )
        with self.assertRaises(ValueError):
            forbidden_bbox_to_label([10, 0, 5, 50], 1000)


class SemanticReductionTests(unittest.TestCase):
    def test_ignores_geometry_classes_and_maps_direct_sign(self):
        observation = resolve_semantic_detections([
            detection("Interact", 1.0, 500),
            detection("Corner", 0.99, 200),
            detection("Right", 0.91, 750),
        ], frame_width=1000)

        self.assertEqual("RIGHT", observation.sign_label)
        self.assertAlmostEqual(0.91, observation.sign_confidence)
        self.assertIsNone(observation.signal_label)

    def test_generic_forbidden_uses_bbox_position(self):
        labels = []
        for centre_x in (200, 500, 800):
            observation = resolve_semantic_detections([
                detection("Forbidden", 0.88, centre_x),
            ], frame_width=1000)
            labels.append(observation.sign_label)
        self.assertEqual(["NO_LEFT", "NO_STRAIGHT", "NO_RIGHT"], labels)

    def test_ambiguous_forbidden_and_close_sign_conflict_emit_no_sign(self):
        ambiguous = resolve_semantic_detections([
            detection("Forbidden", 0.95, 400),
        ], frame_width=1000)
        self.assertIsNone(ambiguous.sign_label)

        conflict = resolve_semantic_detections([
            detection("Left", 0.88, 300),
            detection("Right", 0.82, 700),
        ], frame_width=1000, conflict_margin=0.12)
        self.assertIsNone(conflict.sign_label)

    def test_clear_sign_winner_is_accepted(self):
        observation = resolve_semantic_detections([
            detection("Left", 0.95, 300),
            detection("Right", 0.70, 700),
        ], frame_width=1000, conflict_margin=0.12)
        self.assertEqual("LEFT", observation.sign_label)
        self.assertAlmostEqual(0.95, observation.sign_confidence)

    def test_red_wins_simultaneous_red_and_green(self):
        observation = resolve_semantic_detections([
            detection("Green_Light", 0.99, 500),
            detection("Red_Light", 0.61, 500),
        ], frame_width=1000)
        self.assertEqual("RED", observation.signal_label)
        self.assertAlmostEqual(0.61, observation.signal_confidence)

    def test_low_confidence_detection_is_dropped(self):
        observation = resolve_semantic_detections([
            detection("Green_Light", 0.59, 500),
            detection("Right", 0.59, 700),
        ], frame_width=1000, min_confidence=0.60)
        self.assertIsNone(observation.signal_label)
        self.assertIsNone(observation.sign_label)

    def test_non_finite_confidence_is_dropped(self):
        observation = resolve_semantic_detections([
            detection("Green_Light", float("nan"), 500),
            detection("Red_Light", float("inf"), 500),
        ], frame_width=1000)
        self.assertIsNone(observation.signal_label)


class FakeBoxes(object):
    def __init__(self):
        self.cls = [6, 7, 4]
        self.conf = [0.92, 0.87, 1.0]
        self.xyxy = [
            [450, 10, 490, 60],
            [700, 50, 760, 110],
            [0, 0, 999, 400],
        ]


class FakeResult(object):
    def __init__(self):
        self.names = MODEL_NAMES
        self.boxes = FakeBoxes()


class FakeModel(object):
    def __init__(self):
        self.names = MODEL_NAMES
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [FakeResult()]


class YoloDetectorTests(unittest.TestCase):
    def test_parses_ultralytics_result_without_importing_ultralytics(self):
        model = FakeModel()
        detector = YoloSemanticDetector(
            model=model, min_confidence=0.60, image_size=512, device="cpu"
        )
        frame = np.zeros((480, 1000, 3), dtype=np.uint8)

        observation = detector.detect(frame)

        self.assertEqual("RIGHT", observation.sign_label)
        self.assertEqual("RED", observation.signal_label)
        self.assertGreaterEqual(observation.latency_ms, 0.0)
        self.assertEqual(512, model.kwargs["imgsz"])
        self.assertEqual("cpu", model.kwargs["device"])
        self.assertIs(frame, model.kwargs["source"])

    def test_rejects_checkpoint_with_wrong_class_contract(self):
        model = FakeModel()
        model.names = {0: "red", 1: "green"}
        with self.assertRaises(ValueError):
            YoloSemanticDetector(model=model)


class FakeSemanticDetector(object):
    def detect(self, frame):
        del frame
        return SemanticObservation(
            sign_label="NO_LEFT",
            sign_confidence=0.91,
            latency_ms=5.0,
        )


class FailingSemanticDetector(object):
    def detect(self, frame):
        del frame
        raise RuntimeError("inference failed")


class LatestFrameWorkerTests(unittest.TestCase):
    @staticmethod
    def wait_for_result(worker):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            result, sequence, error = worker.snapshot()
            if sequence > 0:
                return result, sequence, error
            time.sleep(0.005)
        raise AssertionError("semantic worker did not finish")

    def test_worker_attaches_source_identity_and_timestamp(self):
        worker = LatestFrameSemanticWorker(FakeSemanticDetector())
        arrival = time.monotonic()
        try:
            worker.submit(
                np.zeros((4, 6, 3), dtype=np.uint8),
                source_local_seq=7,
                source_arrival_stamp=arrival,
                source_frame_seq=101,
                source_stamp_ns=123456789,
            )
            result, sequence, error = self.wait_for_result(worker)
        finally:
            worker.close()

        self.assertEqual(1, sequence)
        self.assertIsNone(error)
        self.assertEqual("NO_LEFT", result.sign_label)
        self.assertEqual(7, result.source_local_seq)
        self.assertEqual(101, result.source_frame_seq)
        self.assertEqual(123456789, result.source_stamp_ns)
        self.assertEqual(arrival, result.source_arrival_stamp)

    def test_worker_failure_revokes_previous_semantics(self):
        worker = LatestFrameSemanticWorker(FailingSemanticDetector())
        try:
            worker.submit(
                np.zeros((4, 6, 3), dtype=np.uint8),
                source_local_seq=1,
                source_arrival_stamp=time.monotonic(),
            )
            result, sequence, error = self.wait_for_result(worker)
        finally:
            worker.close()

        self.assertEqual(1, sequence)
        self.assertIsNotNone(error)
        self.assertIsNone(result.sign_label)
        self.assertIsNone(result.signal_label)
        self.assertEqual(1, result.source_local_seq)

    def test_worker_rejects_invalid_source_metadata_before_thread(self):
        worker = LatestFrameSemanticWorker(FakeSemanticDetector())
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        try:
            for invalid_seq in (True, 0, 1.5):
                with self.subTest(invalid_seq=invalid_seq):
                    with self.assertRaises(ValueError):
                        worker.submit(frame, invalid_seq, time.monotonic())
            with self.assertRaises(ValueError):
                worker.submit(frame, 1, float("nan"))
            with self.assertRaises(ValueError):
                worker.submit(
                    frame, 1, time.monotonic(), source_stamp_ns=True
                )
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
