# -*- coding: utf-8 -*-
"""Pure tests for runner boundary checks; no ROS or real hardware."""

from __future__ import absolute_import

import sys
import time
import types
import unittest
import json
from unittest import mock

import numpy as np

from src.smart_city.main_smart_city_v2 import (
    NullActuator,
    RosBuffers,
    Runtime,
    VehicleActuator,
    camera_message_identity,
    ros_image_to_bgr,
)
from src.smart_city.v2.config import SmartCityConfig
from src.smart_city.v2.semantic import SemanticObservation


class FakeStamp(object):
    def __init__(self, value):
        self.value = value

    def to_nsec(self):
        return self.value


class FakeHeader(object):
    def __init__(self, seq=0, stamp=0):
        self.seq = seq
        self.stamp = FakeStamp(stamp)


class FakeImageMessage(object):
    def __init__(self, data, width=2, height=1, encoding="bgr8", step=6,
                 seq=0, stamp=0):
        self.data = data
        self.width = width
        self.height = height
        self.encoding = encoding
        self.step = step
        self.header = FakeHeader(seq, stamp)


class FakeCar(object):
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0


class FakeRacerController(object):
    last_instance = None

    def __init__(self):
        self.car = FakeCar()
        self._mock = False
        FakeRacerController.last_instance = self

    def set_steering(self, value):
        self.car.steering = value

    def set_throttle(self, value):
        self.car.throttle = value

    def stop(self):
        self.car.steering = 0.0
        self.car.throttle = 0.0


class FakeCommand(object):
    steering = 0.2
    throttle = 0.1


class FakeTelemetry(object):
    def write(self, *args, **kwargs):
        del args, kwargs


class FakeDetector(object):
    def detect(self, frame):
        del frame
        return SemanticObservation()


class FakeRos(object):
    def __init__(self):
        self.warnings = []

    @staticmethod
    def Subscriber(*args, **kwargs):
        del args, kwargs
        return object()

    @staticmethod
    def Service(*args, **kwargs):
        del args, kwargs
        return object()

    def logwarn_throttle(self, *args):
        self.warnings.append(args)


class RunnerSafetyTests(unittest.TestCase):

    @staticmethod
    def ros_message_modules():
        sensor_package = types.ModuleType("sensor_msgs")
        sensor_messages = types.ModuleType("sensor_msgs.msg")
        sensor_messages.Image = object
        sensor_messages.LaserScan = object
        std_package = types.ModuleType("std_msgs")
        std_messages = types.ModuleType("std_msgs.msg")
        std_messages.Bool = object
        std_messages.String = object
        service_package = types.ModuleType("std_srvs")
        service_module = types.ModuleType("std_srvs.srv")
        service_module.Trigger = object
        service_module.TriggerResponse = object
        return {
            "sensor_msgs": sensor_package,
            "sensor_msgs.msg": sensor_messages,
            "std_msgs": std_package,
            "std_msgs.msg": std_messages,
            "std_srvs": service_package,
            "std_srvs.srv": service_module,
        }

    def test_camera_identity_uses_header_and_checksum_fallback(self):
        first = FakeImageMessage(b"\x00\x01\x02\x03\x04\x05", seq=4, stamp=10)
        replay = FakeImageMessage(b"different", seq=4, stamp=10)
        next_frame = FakeImageMessage(b"same", seq=5, stamp=11)
        self.assertEqual(camera_message_identity(first),
                         camera_message_identity(replay))
        self.assertNotEqual(camera_message_identity(first),
                            camera_message_identity(next_frame))

        zero_a = FakeImageMessage(b"\x00\x01\x02\x03\x04\x05")
        zero_b = FakeImageMessage(b"\x00\x01\x02\x03\x04\x06")
        self.assertNotEqual(camera_message_identity(zero_a),
                            camera_message_identity(zero_b))

    def test_ros_image_decoder_respects_stride(self):
        message = FakeImageMessage(
            bytes((1, 2, 3, 4, 5, 6, 99, 99)),
            width=2,
            height=1,
            encoding="bgr8",
            step=8,
        )
        frame = ros_image_to_bgr(message)
        self.assertEqual((1, 2, 3), tuple(frame[0, 0]))
        self.assertEqual((4, 5, 6), tuple(frame[0, 1]))

    def test_config_rejects_nan_and_live_requires_signed_calibration(self):
        config = SmartCityConfig()
        with self.assertRaises(ValueError):
            config.update({"cruise_throttle": float("nan")})
        with self.assertRaises((TypeError, ValueError)):
            config.update({"actuator_watchdog_seconds": "NaN"})
        self.assertEqual(0.20, config.actuator_watchdog_seconds)
        with self.assertRaises(ValueError):
            config.validate_live()

        config.update({"calibrated": True, "calibration_id": "bench-001"})
        self.assertIs(config, config.validate_live())
        config.update({"turn_max_seconds": 20.0})
        with self.assertRaises(ValueError):
            config.validate_live()

        for override in (
            {"stop_hold_seconds": -1.0},
            {"green_danger_ratio": 2.0},
            {"lidar_stop_distance_m": -1.0},
        ):
            fresh = SmartCityConfig()
            with self.assertRaises(ValueError):
                fresh.update(override)

        for override in (
            {"stop_confirm_frames": 1},
            {"initial_lane_stable_frames": 1},
            {"ai_confirm_frames": 1},
            {"nudge_left_seconds": 999.0},
        ):
            fresh = SmartCityConfig()
            fresh.update({
                "calibrated": True,
                "calibration_id": "bench-001",
            })
            fresh.update(override)
            with self.assertRaises(ValueError):
                fresh.validate_live()

    def test_actuator_watchdog_stops_and_latches(self):
        fake_module = types.ModuleType("src.core.control.racer_controller")
        fake_module.RacerController = FakeRacerController
        config = SmartCityConfig()
        config.actuator_watchdog_seconds = 0.04

        with mock.patch.dict(sys.modules, {
            "src.core.control.racer_controller": fake_module,
        }):
            actuator = VehicleActuator(config)
            try:
                actuator.apply(FakeCommand())
                self.assertGreater(
                    FakeRacerController.last_instance.car.throttle, 0.0
                )
                time.sleep(0.10)
                self.assertEqual(
                    0.0, FakeRacerController.last_instance.car.throttle
                )
                with self.assertRaises(RuntimeError):
                    actuator.apply(FakeCommand())
            finally:
                actuator.close()

    def test_actuator_emergency_stop_is_atomic_and_latched(self):
        fake_module = types.ModuleType("src.core.control.racer_controller")
        fake_module.RacerController = FakeRacerController
        config = SmartCityConfig()
        with mock.patch.dict(sys.modules, {
            "src.core.control.racer_controller": fake_module,
        }):
            actuator = VehicleActuator(config)
            try:
                actuator.apply(FakeCommand())
                actuator.emergency_stop()
                self.assertEqual(
                    0.0, FakeRacerController.last_instance.car.throttle
                )
                with self.assertRaises(RuntimeError):
                    actuator.apply(FakeCommand())
            finally:
                actuator.close()

    def test_runtime_pairs_external_semantics_with_its_own_sequence(self):
        runtime = Runtime(
            SmartCityConfig(),
            {"intersections": []},
            NullActuator(),
            FakeTelemetry(),
            FakeDetector(),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = SemanticObservation(
            signal_label="RED", signal_confidence=0.99, source_local_seq=10
        )
        second = SemanticObservation(
            signal_label="GREEN", signal_confidence=0.99, source_local_seq=11
        )
        runtime.step(
            frame, now=1.0, seq=5, external_semantic=first, semantic_seq=20
        )
        runtime.step(
            frame, now=1.1, seq=5, external_semantic=second, semantic_seq=21
        )
        self.assertIs(runtime.last_semantic, second)
        self.assertEqual("GREEN", runtime.last_semantic.signal_label)

    def test_ros_semantic_requires_recent_ordered_camera_identity(self):
        config = SmartCityConfig()
        fake_ros = FakeRos()
        with mock.patch.dict(sys.modules, self.ros_message_modules()):
            buffers = RosBuffers(
                fake_ros, config, use_lidar=False,
                semantic_topic="/smart_city/semantic",
            )

        camera = FakeImageMessage(
            bytes((1, 2, 3, 4, 5, 6)), seq=9, stamp=1234567890123456789
        )
        buffers._camera_callback(camera)
        semantic_message = types.SimpleNamespace(data=json.dumps({
            "signal_label": "GREEN",
            "signal_confidence": 0.99,
            "source_frame_seq": 9,
            "source_stamp_ns": 1234567890123456789,
        }))
        buffers._semantic_callback(semantic_message)
        self.assertEqual(1, buffers.semantic_seq)
        self.assertEqual(1, buffers.semantic.source_local_seq)

        buffers._semantic_callback(semantic_message)
        self.assertEqual(1, buffers.semantic_seq)
        missing_source = types.SimpleNamespace(data=json.dumps({
            "signal_label": "GREEN", "signal_confidence": 0.99,
        }))
        buffers._semantic_callback(missing_source)
        self.assertEqual(1, buffers.semantic_seq)


if __name__ == "__main__":
    unittest.main()
