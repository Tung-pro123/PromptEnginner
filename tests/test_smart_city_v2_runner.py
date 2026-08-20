# -*- coding: utf-8 -*-
"""Pure tests for runner boundary checks; no ROS or real hardware."""

from __future__ import absolute_import

import sys
import os
import tempfile
import threading
import time
import types
import unittest
import json
from unittest import mock

import numpy as np

from src.smart_city.main_smart_city_v2 import (
    DEFAULT_SCENARIO,
    NullActuator,
    RosBuffers,
    Runtime,
    VehicleActuator,
    camera_message_identity,
    ros_image_to_bgr,
    should_arm_runtime,
    validate_live_inputs,
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


class FakeMockRacerController(FakeRacerController):
    def __init__(self):
        super(FakeMockRacerController, self).__init__()
        self._mock = True


class FakeRacerWithoutHardware(object):
    def __init__(self):
        self.car = object()
        self._mock = False

    def stop(self):
        pass


class BlockingFakeRacerController(FakeRacerController):
    steering_started = threading.Event()
    release_steering = threading.Event()

    def set_steering(self, value):
        self.steering_started.set()
        if not self.release_steering.wait(1.0):
            raise RuntimeError("test timed out waiting to release steering")
        super(BlockingFakeRacerController, self).set_steering(value)


class FailingThrottleRacerController(FakeRacerController):
    def set_throttle(self, value):
        if value > 0.0:
            raise IOError("simulated motor bus failure")
        super(FailingThrottleRacerController, self).set_throttle(value)


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
        config.update({"semantic_ttl_seconds": 0.51})
        with self.assertRaises(ValueError):
            config.validate_live()
        config.update({"semantic_ttl_seconds": 0.35})
        config.update({"turn_max_seconds": 20.0})
        with self.assertRaises(ValueError):
            config.validate_live()

        for override in (
            {"stop_hold_seconds": -1.0},
            {"green_danger_ratio": 2.0},
            {"lidar_stop_distance_m": -1.0},
            {"nudge_left_seconds": 999.0},
        ):
            fresh = SmartCityConfig()
            with self.assertRaises(ValueError):
                fresh.update(override)

        for override in (
            {"stop_confirm_frames": 1},
            {"initial_lane_stable_frames": 1},
            {"ai_confirm_frames": 1},
            {"nudge_marker_clear_frames": 1},
            {"turn_steering_right": -0.76},
            {"lane_min_confidence": 0.0},
            {"reacquire_center_error_ratio": 1.0},
            {"reacquire_heading_error_ratio": 1.0},
            {"exit_clear_frames": 1},
            {"intersection_cooldown_seconds": 0.0},
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

    def test_pending_apply_cannot_reenergise_after_emergency_stop(self):
        fake_module = types.ModuleType("src.core.control.racer_controller")
        BlockingFakeRacerController.steering_started.clear()
        BlockingFakeRacerController.release_steering.clear()
        fake_module.RacerController = BlockingFakeRacerController
        config = SmartCityConfig()
        errors = []

        with mock.patch.dict(sys.modules, {
            "src.core.control.racer_controller": fake_module,
        }):
            actuator = VehicleActuator(config)
            apply_thread = threading.Thread(
                target=lambda: self._capture_error(
                    errors, actuator.apply, FakeCommand()
                )
            )
            stop_thread = threading.Thread(
                target=lambda: self._capture_error(
                    errors, actuator.emergency_stop
                )
            )
            try:
                apply_thread.start()
                self.assertTrue(
                    BlockingFakeRacerController.steering_started.wait(0.5)
                )
                stop_thread.start()
                BlockingFakeRacerController.release_steering.set()
                apply_thread.join(1.0)
                stop_thread.join(1.0)
                self.assertFalse(apply_thread.is_alive())
                self.assertFalse(stop_thread.is_alive())
                self.assertFalse(errors)
                self.assertEqual(
                    0.0, BlockingFakeRacerController.last_instance.car.throttle
                )
                with self.assertRaises(RuntimeError):
                    actuator.apply(FakeCommand())
            finally:
                BlockingFakeRacerController.release_steering.set()
                actuator.close()

    def test_actuator_close_is_latched_and_live_refuses_fake_hardware(self):
        fake_module = types.ModuleType("src.core.control.racer_controller")
        fake_module.RacerController = FakeRacerController
        config = SmartCityConfig()
        with mock.patch.dict(sys.modules, {
            "src.core.control.racer_controller": fake_module,
        }):
            actuator = VehicleActuator(config)
            actuator.close()
            with self.assertRaises(RuntimeError):
                actuator.apply(FakeCommand())
            self.assertEqual(0.0, FakeRacerController.last_instance.car.throttle)

        for fake_type in (FakeMockRacerController, FakeRacerWithoutHardware):
            fake_module.RacerController = fake_type
            with mock.patch.dict(sys.modules, {
                "src.core.control.racer_controller": fake_module,
            }):
                with self.assertRaises(RuntimeError):
                    VehicleActuator(config)

    def test_hardware_write_failure_stops_and_latches_actuator(self):
        fake_module = types.ModuleType("src.core.control.racer_controller")
        fake_module.RacerController = FailingThrottleRacerController
        config = SmartCityConfig()
        with mock.patch.dict(sys.modules, {
            "src.core.control.racer_controller": fake_module,
        }):
            actuator = VehicleActuator(config)
            try:
                with self.assertRaises(RuntimeError):
                    actuator.apply(FakeCommand())
                self.assertEqual(0.0, FakeRacerController.last_instance.car.throttle)
                self.assertEqual(0.0, FakeRacerController.last_instance.car.steering)
                with self.assertRaises(RuntimeError):
                    actuator.apply(FakeCommand())
            finally:
                actuator.close()

    def test_live_validation_requires_calibration_route_and_lidar(self):
        config = SmartCityConfig()
        config.update({"calibrated": True, "calibration_id": "bench-001"})
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "calibration.json")
            scenario_path = os.path.join(directory, "route.json")
            with open(config_path, "w", encoding="utf-8") as output:
                json.dump({"calibrated": True}, output)
            with open(scenario_path, "w", encoding="utf-8") as output:
                json.dump({
                    "validated_for_live": True,
                    "route_id": "official-course-a",
                    "intersections": [],
                }, output)
            args = types.SimpleNamespace(
                config=config_path,
                scenario=scenario_path,
                use_lidar=True,
                semantic_topic="/smart_city/semantic",
                require_ai=True,
                mock_sign=None,
                mock_signal=None,
            )
            self.assertIsNone(validate_live_inputs(args, config))

            args.semantic_topic = None
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)
            args.semantic_topic = "/smart_city/semantic"
            args.require_ai = False
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)
            args.require_ai = True
            args.mock_signal = "GREEN"
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)
            args.mock_signal = None

            with open(scenario_path, "w", encoding="utf-8") as output:
                json.dump({
                    "validated_for_live": True,
                    "route_id": "official-course-a",
                    "intersections": [{
                        "id": "I01",
                        "allowed": ["LEFT", "RIGHT"],
                        "requires_sign": True,
                        "mock_sign": "NO_LEFT",
                    }],
                }, output)
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)
            with open(scenario_path, "w", encoding="utf-8") as output:
                json.dump({
                    "validated_for_live": True,
                    "route_id": "official-course-a",
                    "intersections": [{
                        "id": "I01",
                        "allowed": ["LEFT", "RIGHT"],
                        "action": "RIGHT",
                        "requires_sign": "true",
                    }],
                }, output)
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)
            with open(scenario_path, "w", encoding="utf-8") as output:
                json.dump({
                    "validated_for_live": True,
                    "route_id": "official-course-a",
                    "intersections": [],
                }, output)

            args.use_lidar = False
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)
            args.use_lidar = True
            args.config = None
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)
            args.config = config_path
            args.scenario = DEFAULT_SCENARIO
            with self.assertRaises(ValueError):
                validate_live_inputs(args, config)

    def test_motor_mode_never_auto_arms_from_startup_flag(self):
        self.assertTrue(
            should_arm_runtime(False, 0, None, 10.0, 0.75, False)
        )
        self.assertFalse(
            should_arm_runtime(True, 0, None, 10.0, 0.75, False)
        )
        self.assertTrue(
            should_arm_runtime(True, 1, 9.5, 10.0, 0.75, False)
        )
        self.assertFalse(
            should_arm_runtime(True, 1, 8.0, 10.0, 0.75, False)
        )
        self.assertFalse(
            should_arm_runtime(True, 1, 9.5, 10.0, 0.75, True)
        )

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
            signal_label="RED", signal_confidence=0.99,
            stamp=1.0, source_frame_seq=104, source_stamp_ns=1004,
            source_local_seq=4, source_arrival_stamp=0.95,
        )
        second = SemanticObservation(
            signal_label="GREEN", signal_confidence=0.99,
            stamp=1.1, source_frame_seq=105, source_stamp_ns=1005,
            source_local_seq=5, source_arrival_stamp=1.05,
        )
        runtime.step(
            frame, now=1.0, seq=5, external_semantic=first, semantic_seq=20
        )
        runtime.step(
            frame, now=1.1, seq=5, external_semantic=second, semantic_seq=21
        )
        self.assertIs(runtime.last_semantic, second)
        self.assertEqual("GREEN", runtime.last_semantic.signal_label)

    def test_runtime_rejects_stale_future_and_out_of_order_semantics(self):
        runtime = Runtime(
            SmartCityConfig(),
            {"intersections": []},
            NullActuator(),
            FakeTelemetry(),
            FakeDetector(),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        accepted = SemanticObservation(
            signal_label="GREEN", signal_confidence=0.99,
            stamp=1.0, source_frame_seq=1, source_stamp_ns=1001,
            source_local_seq=1, source_arrival_stamp=0.95,
        )
        runtime.step(
            frame, now=1.0, seq=2,
            external_semantic=accepted, semantic_seq=10,
        )
        self.assertEqual("GREEN", runtime.last_semantic.signal_label)

        future = SemanticObservation(
            signal_label="GREEN", signal_confidence=0.99,
            stamp=1.1, source_frame_seq=3, source_stamp_ns=1003,
            source_local_seq=3, source_arrival_stamp=1.05,
        )
        runtime.step(
            frame, now=1.1, seq=2,
            external_semantic=future, semantic_seq=11,
        )
        self.assertIsNone(runtime.last_semantic.signal_label)

        fresh = SemanticObservation(
            signal_label="RED", signal_confidence=0.99,
            stamp=1.2, source_frame_seq=2, source_stamp_ns=1002,
            source_local_seq=2, source_arrival_stamp=1.15,
        )
        runtime.step(
            frame, now=1.2, seq=3,
            external_semantic=fresh, semantic_seq=12,
        )
        self.assertEqual("RED", runtime.last_semantic.signal_label)

        stale = SemanticObservation(
            signal_label="GREEN", signal_confidence=0.99,
            stamp=1.0, source_frame_seq=3, source_stamp_ns=1003,
            source_local_seq=3, source_arrival_stamp=1.0,
        )
        runtime.step(
            frame, now=3.0, seq=4,
            external_semantic=stale, semantic_seq=13,
        )
        self.assertIsNone(runtime.last_semantic.signal_label)

        out_of_order = SemanticObservation(
            signal_label="GREEN", signal_confidence=0.99,
            stamp=3.0, source_frame_seq=4, source_stamp_ns=1004,
            source_local_seq=4, source_arrival_stamp=2.95,
        )
        runtime.step(
            frame, now=3.0, seq=4,
            external_semantic=out_of_order, semantic_seq=12,
        )
        self.assertIsNone(runtime.last_semantic.signal_label)

    def test_runtime_ttl_is_measured_from_source_frame_not_ai_callback(self):
        config = SmartCityConfig()
        runtime = Runtime(
            config,
            {"intersections": []},
            NullActuator(),
            FakeTelemetry(),
            FakeDetector(),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # The AI callback arrived while the source was still inside TTL, but
        # retaining that result for another callback-sized TTL would make the
        # original camera frame stale.  Runtime must use total source age.
        delayed = SemanticObservation(
            signal_label="GREEN",
            signal_confidence=0.99,
            stamp=1.30,
            source_frame_seq=1,
            source_stamp_ns=1001,
            source_local_seq=1,
            source_arrival_stamp=1.00,
        )
        runtime.step(
            frame,
            now=1.50,
            seq=1,
            external_semantic=delayed,
            semantic_seq=1,
        )
        self.assertIsNone(runtime.last_semantic.signal_label)

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
        self.assertEqual(
            buffers._camera_history[-1]["arrival"],
            buffers.semantic.source_arrival_stamp,
        )

        buffers._semantic_callback(semantic_message)
        self.assertEqual(2, buffers.semantic_seq)
        self.assertIsNone(buffers.semantic.signal_label)
        self.assertIsNone(buffers.semantic_stamp)
        missing_source = types.SimpleNamespace(data=json.dumps({
            "signal_label": "GREEN", "signal_confidence": 0.99,
        }))
        buffers._semantic_callback(missing_source)
        self.assertEqual(3, buffers.semantic_seq)
        self.assertIsNone(buffers.semantic.signal_label)

        forbidden_geometry = types.SimpleNamespace(data=json.dumps({
            "signal_label": "GREEN", "signal_confidence": 0.99,
            "crosswalk_conf": 0.99,
            "source_frame_seq": 9,
            "source_stamp_ns": 1234567890123456789,
        }))
        buffers._semantic_callback(forbidden_geometry)
        self.assertEqual(4, buffers.semantic_seq)
        self.assertIsNone(buffers.semantic.signal_label)

    def test_ros_semantic_mismatch_and_stale_source_revoke_green(self):
        config = SmartCityConfig()
        fake_ros = FakeRos()
        with mock.patch.dict(sys.modules, self.ros_message_modules()):
            buffers = RosBuffers(
                fake_ros, config, use_lidar=False,
                semantic_topic="/smart_city/semantic",
            )

        first_stamp = 1234567890123456700
        buffers._camera_callback(FakeImageMessage(
            bytes((1, 2, 3, 4, 5, 6)), seq=20, stamp=first_stamp
        ))
        first_green = types.SimpleNamespace(data=json.dumps({
            "signal_label": "GREEN",
            "signal_confidence": 0.99,
            "source_frame_seq": 20,
            "source_stamp_ns": first_stamp,
        }))
        buffers._semantic_callback(first_green)
        self.assertEqual("GREEN", buffers.semantic.signal_label)

        mismatched_stamp = types.SimpleNamespace(data=json.dumps({
            "signal_label": "GREEN",
            "signal_confidence": 0.99,
            "source_frame_seq": 20,
            "source_stamp_ns": first_stamp + 1,
        }))
        buffers._semantic_callback(mismatched_stamp)
        self.assertIsNone(buffers.semantic.signal_label)
        self.assertIsNone(buffers.semantic_stamp)

        second_stamp = first_stamp + 100
        buffers._camera_callback(FakeImageMessage(
            bytes((6, 5, 4, 3, 2, 1)), seq=21, stamp=second_stamp
        ))
        second_green = types.SimpleNamespace(data=json.dumps({
            "signal_label": "GREEN",
            "signal_confidence": 0.99,
            "source_frame_seq": 21,
            "source_stamp_ns": second_stamp,
        }))
        buffers._semantic_callback(second_green)
        self.assertEqual("GREEN", buffers.semantic.signal_label)
        with buffers.lock:
            buffers._camera_history[-1]["arrival"] -= (
                config.semantic_ttl_seconds + 0.1
            )
        buffers._semantic_callback(second_green)
        self.assertIsNone(buffers.semantic.signal_label)
        self.assertIsNone(buffers.semantic_stamp)

    @staticmethod
    def _capture_error(errors, function, *args):
        try:
            function(*args)
        except Exception as exc:  # pragma: no cover - assertion reports value
            errors.append(exc)


if __name__ == "__main__":
    unittest.main()
