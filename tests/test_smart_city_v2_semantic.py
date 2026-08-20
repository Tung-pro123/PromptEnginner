# -*- coding: utf-8 -*-
"""Contract tests for the Smart City V2 semantic-detector boundary."""

from __future__ import division

import math
import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.smart_city.v2.semantic import (  # noqa: E402
    FunctionSemanticDetector,
    ManualSemanticDetector,
    NullSemanticDetector,
    SemanticObservation,
)


class SemanticObservationTests(unittest.TestCase):
    def test_normalises_labels_and_preserves_confidence(self):
        observation = SemanticObservation(
            sign_label="  no-left  ",
            sign_confidence="0.75",
            signal_label="red light",
            signal_confidence=1,
            stamp=12.5,
        )

        self.assertEqual(observation.sign_label, "NO_LEFT")
        self.assertEqual(observation.signal_label, "RED_LIGHT")
        self.assertAlmostEqual(observation.sign_confidence, 0.75)
        self.assertEqual(observation.signal_confidence, 1.0)
        self.assertEqual(observation.stamp, 12.5)
        self.assertEqual(observation.as_dict()["sign_label"], "NO_LEFT")

    def test_empty_labels_become_none_and_confidence_boundaries_are_valid(self):
        observation = SemanticObservation(
            sign_label="   ",
            sign_confidence=0.0,
            signal_label=None,
            signal_confidence=1.0,
        )

        self.assertIsNone(observation.sign_label)
        self.assertIsNone(observation.signal_label)
        self.assertEqual(observation.sign_confidence, 0.0)
        self.assertEqual(observation.signal_confidence, 1.0)
        self.assertTrue(math.isfinite(observation.stamp))

    def test_rejects_invalid_label_and_out_of_range_confidence(self):
        with self.assertRaises(TypeError):
            SemanticObservation(sign_label=123)
        for confidence in (-0.001, 1.001, float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    SemanticObservation(sign_confidence=confidence)

    def test_source_identity_preserves_large_ros_nanoseconds(self):
        stamp_ns = 1787049300123456789
        observation = SemanticObservation(
            source_frame_seq=1842,
            source_stamp_ns=stamp_ns,
            source_local_seq=77,
        )
        self.assertEqual(stamp_ns, observation.source_stamp_ns)
        with self.assertRaises(TypeError):
            SemanticObservation(source_stamp_ns=float(stamp_ns))


class FunctionSemanticDetectorTests(unittest.TestCase):
    def test_accepts_semantic_dictionary_and_passes_frame_through(self):
        received_frames = []

        def model(frame):
            received_frames.append(frame)
            return {
                "sign_label": "ban right",
                "sign_confidence": 0.83,
                "signal_label": "green-light",
                "signal_confidence": 0.91,
            }

        detector = FunctionSemanticDetector(model)
        frame = object()
        observation = detector.detect(frame)

        self.assertEqual(received_frames, [frame])
        self.assertIsInstance(observation, SemanticObservation)
        self.assertEqual(observation.sign_label, "BAN_RIGHT")
        self.assertAlmostEqual(observation.sign_confidence, 0.83)
        self.assertEqual(observation.signal_label, "GREEN_LIGHT")
        self.assertAlmostEqual(observation.signal_confidence, 0.91)
        self.assertIsNotNone(observation.latency_ms)
        self.assertGreaterEqual(observation.latency_ms, 0.0)

    def test_rejects_each_control_output_key(self):
        for forbidden_key in ("steering", "throttle", "action"):
            with self.subTest(forbidden_key=forbidden_key):
                detector = FunctionSemanticDetector(
                    lambda _frame, key=forbidden_key: {
                        "sign_label": "STOP",
                        key: 0.5,
                    }
                )
                with self.assertRaises(ValueError) as context:
                    detector.detect(object())
                self.assertIn(forbidden_key, str(context.exception))

    def test_rejects_non_callable_and_non_semantic_output(self):
        with self.assertRaises(TypeError):
            FunctionSemanticDetector(None)

        detector = FunctionSemanticDetector(lambda _frame: "TURN_LEFT")
        with self.assertRaises(TypeError):
            detector.detect(object())

    def test_accepts_semantic_observation_from_function(self):
        original = SemanticObservation(sign_label="stop", sign_confidence=0.9)
        detector = FunctionSemanticDetector(lambda _frame: original)

        result = detector.detect(object())

        self.assertIs(result, original)
        self.assertEqual(result.sign_label, "STOP")
        self.assertIsNotNone(result.latency_ms)


class ManualSemanticDetectorTests(unittest.TestCase):
    def test_manual_sign_and_signal_keep_independent_confidences(self):
        detector = ManualSemanticDetector()
        detector.set_sign("stop", confidence=0.81)
        detector.set_signal("red", confidence=0.63)

        observation = detector.detect(object())

        self.assertAlmostEqual(observation.sign_confidence, 0.81)
        self.assertAlmostEqual(observation.signal_confidence, 0.63)

    def test_manual_clear_removes_labels_and_confidences(self):
        detector = ManualSemanticDetector(
            sign_label="no left", signal_label="red", confidence=0.88
        )
        before = detector.detect(object())
        self.assertEqual(before.sign_label, "NO_LEFT")
        self.assertEqual(before.signal_label, "RED")
        self.assertAlmostEqual(before.sign_confidence, 0.88)
        self.assertAlmostEqual(before.signal_confidence, 0.88)

        detector.clear()
        after = detector.detect(object())

        self.assertIsNone(after.sign_label)
        self.assertIsNone(after.signal_label)
        self.assertIsNone(after.sign_confidence)
        self.assertIsNone(after.signal_confidence)

    def test_null_detector_returns_empty_observation(self):
        observation = NullSemanticDetector().detect(object())
        self.assertIsInstance(observation, SemanticObservation)
        self.assertIsNone(observation.sign_label)
        self.assertIsNone(observation.signal_label)


if __name__ == "__main__":
    unittest.main()
