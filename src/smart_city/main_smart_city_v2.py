#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe Smart City V2 runner for offline replay and ROS shadow/live mode.

Examples are documented in ``docs/SMART_CITY_V2_GUIDE.md``.  Motor output is
disabled by default.  Live movement requires both ``--enable-motors`` and
``--arm`` so an accidental invocation cannot move the car.
"""

from __future__ import absolute_import, division, print_function

import sys
# Prevent ROS Melodic Python 2.7 dist-packages from breaking Python 3 stdlib/cv2 imports
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2
import argparse
import csv
import datetime
import json
import math
import os
import threading
import time
import zlib

import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.smart_city.v2.config import SmartCityConfig
from src.smart_city.v2.controller import DriveCommand, SmartCityFSM, SmartCityState
from src.smart_city.v2.decision import ScenarioDecisionProvider
from src.smart_city.v2.perception import SmartCityPerception
from src.smart_city.v2.semantic import ManualSemanticDetector, SemanticObservation
from src.smart_city.v2.yolo_semantic import (
    LatestFrameSemanticWorker,
    YoloSemanticDetector,
)


DEFAULT_SCENARIO = os.path.join(
    PROJECT_ROOT, "src", "smart_city", "v2", "scenario_example.json"
)


class NullActuator(object):
    """Shadow-mode actuator; retains the intended command for inspection."""

    def __init__(self):
        self.last_command = None

    def apply(self, command):
        self.last_command = command

    def stop(self):
        self.last_command = None

    def emergency_stop(self):
        self.stop()

    def close(self):
        self.stop()


class VehicleActuator(object):
    """Hardware adapter with an independent, latched command watchdog."""

    HARD_MAX_THROTTLE = 0.30
    HARD_MAX_STEERING = 0.95

    def __init__(self, config):
        from src.core.control.racer_controller import RacerController
        self.racer = RacerController()
        if getattr(self.racer, "_mock", False):
            self.racer.stop()
            raise RuntimeError("motor mode refused: RacerController is mocked")
        car = getattr(self.racer, "car", None)
        if not (
            car is not None
            and hasattr(car, "steering")
            and hasattr(car, "throttle")
        ):
            self.racer.stop()
            raise RuntimeError(
                "motor mode requires hardware with steering and throttle"
            )
        self.timeout = float(config.actuator_watchdog_seconds)
        self._lock = threading.Lock()
        self._last_heartbeat = time.monotonic()
        self._moving = False
        self._closed = False
        self._watchdog_tripped = False
        self._emergency_stop_latched = False
        self.racer.stop()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop)
        self._watchdog_thread.daemon = True
        self._watchdog_thread.start()

    def apply(self, command):
        steering = float(command.steering)
        throttle = float(command.throttle)
        if not math.isfinite(steering) or not math.isfinite(throttle):
            self.stop()
            raise RuntimeError("non-finite actuator command")
        if (
            abs(steering) > self.HARD_MAX_STEERING
            or throttle < 0.0
            or throttle > self.HARD_MAX_THROTTLE
        ):
            self.stop()
            raise RuntimeError("actuator command exceeds independent hard cap")
        with self._lock:
            if self._closed:
                self.racer.stop()
                raise RuntimeError("actuator is closed")
            if self._emergency_stop_latched:
                self.racer.stop()
                raise RuntimeError("emergency stop is latched; restart required")
            if self._watchdog_tripped:
                self.racer.stop()
                raise RuntimeError("actuator watchdog is latched; restart required")
            self._last_heartbeat = time.monotonic()
            if throttle <= 0.0:
                self.racer.stop()
                self._moving = False
                return
            # Avoid relying on the uncommitted convenience ``steer`` method.
            try:
                self.racer.set_steering(steering)
                self.racer.set_throttle(throttle)
            except Exception as exc:
                # A partial hardware write must never leave the previous
                # positive throttle active.  Latch the actuator and require a
                # process restart after attempting an immediate zero command.
                self._emergency_stop_latched = True
                self._moving = False
                self._last_heartbeat = time.monotonic()
                try:
                    self.racer.stop()
                except Exception:
                    pass
                raise RuntimeError("hardware actuator write failed") from exc
            self._moving = True

    def stop(self):
        with self._lock:
            self.racer.stop()
            self._moving = False
            self._last_heartbeat = time.monotonic()

    def emergency_stop(self):
        """Atomically stop and reject every later command until restart."""
        with self._lock:
            self._emergency_stop_latched = True
            self.racer.stop()
            self._moving = False
            self._last_heartbeat = time.monotonic()

    def close(self):
        # Closing and stopping must be one critical section.  Calling stop()
        # first leaves a gap in which an already-waiting apply() can energise
        # the motor immediately before _closed is set.
        with self._lock:
            self._closed = True
            self.racer.stop()
            self._moving = False
        if self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=0.35)

    def _watchdog_loop(self):
        interval = max(0.01, min(0.05, self.timeout * 0.25))
        while True:
            time.sleep(interval)
            with self._lock:
                if self._closed:
                    return
                if (
                    self._moving
                    and time.monotonic() - self._last_heartbeat > self.timeout
                ):
                    self.racer.stop()
                    self._moving = False
                    self._watchdog_tripped = True


class CsvTelemetry(object):
    FIELDS = (
        "wall_time",
        "monotonic_s",
        "frame_seq",
        "state",
        "reason",
        "intersection_id",
        "action",
        "steering",
        "throttle",
        "lane_confidence",
        "lane_x_near",
        "lane_x_far",
        "stop_line",
        "stop_line_score",
        "stop_line_y",
        "green_ahead_ratio",
        "green_left_ratio",
        "green_right_ratio",
        "sign_label",
        "sign_confidence",
        "signal_label",
        "signal_confidence",
        "ai_latency_ms",
        "camera_age_ms",
        "front_distance_m",
        "loop_latency_ms",
    )

    def __init__(self, path=None):
        self.handle = None
        self.writer = None
        if path:
            parent = os.path.dirname(os.path.abspath(path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            self.handle = open(path, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
            self.writer.writeheader()

    def write(self, now, seq, command, perception, semantic, camera_age,
              front_distance, loop_latency_ms):
        if self.writer is None:
            return
        row = {
            "wall_time": datetime.datetime.now().isoformat(),
            "monotonic_s": "%.6f" % now,
            "frame_seq": seq,
            "state": command.state,
            "reason": command.reason,
            "intersection_id": command.intersection_id or "",
            "action": command.action or "",
            "steering": "%.5f" % command.steering,
            "throttle": "%.5f" % command.throttle,
            "lane_confidence": "%.5f" % perception.lane_confidence,
            "lane_x_near": _optional_number(perception.lane_x_near),
            "lane_x_far": _optional_number(perception.lane_x_far),
            "stop_line": int(perception.stop_line),
            "stop_line_score": "%.5f" % perception.stop_line_score,
            "stop_line_y": _optional_number(perception.stop_line_y),
            "green_ahead_ratio": "%.5f" % perception.green_ahead_ratio,
            "green_left_ratio": "%.5f" % perception.green_left_ratio,
            "green_right_ratio": "%.5f" % perception.green_right_ratio,
            "sign_label": semantic.sign_label or "",
            "sign_confidence": _optional_number(semantic.sign_confidence),
            "signal_label": semantic.signal_label or "",
            "signal_confidence": _optional_number(semantic.signal_confidence),
            "ai_latency_ms": _optional_number(semantic.latency_ms),
            "camera_age_ms": "%.3f" % (camera_age * 1000.0),
            "front_distance_m": _optional_number(front_distance),
            "loop_latency_ms": "%.3f" % loop_latency_ms,
        }
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class RosBuffers(object):
    """Latest-only ROS buffers; callbacks never run perception or control."""

    SEMANTIC_KEYS = frozenset((
        "sign_label",
        "sign_confidence",
        "signal_label",
        "signal_confidence",
        "latency_ms",
        "source_frame_seq",
        "source_stamp_ns",
        "source_crc32",
    ))

    def __init__(self, rospy, config, use_lidar=False, semantic_topic=None,
                 emergency_stop_callback=None):
        from sensor_msgs.msg import Image, LaserScan
        from std_msgs.msg import Bool
        from std_srvs.srv import Trigger, TriggerResponse

        self.rospy = rospy
        self.config = config
        self.use_lidar = bool(use_lidar)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_stamp = None
        self.frame_seq = 0
        self._last_camera_key = None
        self._last_header_stamp_ns = None
        self._last_header_seq = None
        self._last_content_crc = None
        self._camera_history = []
        self._last_semantic_source_local_seq = None
        self.front_distance = None
        self.lidar_stamp = None
        self.semantic = SemanticObservation()
        self.semantic_stamp = None
        self.semantic_seq = 0
        self.arm_requests = 0
        self.arm_request_stamp = None
        self.estop_latched = False
        self.emergency_stop_callback = emergency_stop_callback

        self.camera_sub = rospy.Subscriber(
            config.camera_topic, Image, self._camera_callback, queue_size=1
        )
        self.lidar_sub = None
        if use_lidar:
            self.lidar_sub = rospy.Subscriber(
                config.lidar_topic, LaserScan, self._lidar_callback, queue_size=1
            )
        self.semantic_sub = None
        if semantic_topic:
            from std_msgs.msg import String
            self.semantic_sub = rospy.Subscriber(
                semantic_topic, String, self._semantic_callback, queue_size=1
            )
        self._trigger_response_type = TriggerResponse
        self.arm_service = rospy.Service(
            config.arm_topic, Trigger, self._arm_service
        )
        self.estop_sub = rospy.Subscriber(
            config.estop_topic, Bool, self._estop_callback, queue_size=1
        )

    def _camera_callback(self, message):
        identity = camera_message_identity(message)
        content_crc = (
            identity[-1]
            if identity[0] == "sample"
            else camera_content_crc(message)
        )
        arrival = time.monotonic()
        with self.lock:
            if not self._camera_sample_is_new_locked(identity, content_crc):
                return
        try:
            frame = ros_image_to_bgr(message)
        except (ValueError, TypeError) as exc:
            self.rospy.logwarn_throttle(2.0, "Camera decode failed: %s" % exc)
            return
        with self.lock:
            # ROS callbacks normally arrive in order, but decoding happens
            # outside the lock.  Re-check here so a slow older decode cannot
            # overwrite a newer frame and corrupt semantic/source pairing.
            if not self._camera_sample_is_new_locked(identity, content_crc):
                return
            self.frame = frame.copy()
            self.frame_stamp = arrival
            self.frame_seq += 1
            self._last_camera_key = identity
            self._last_content_crc = content_crc
            if identity[0] == "header":
                self._last_header_seq = identity[1]
                self._last_header_stamp_ns = identity[2]
            self._camera_history.append({
                "local_seq": self.frame_seq,
                "header_seq": identity[1] if identity[0] == "header" else 0,
                "stamp_ns": identity[2] if identity[0] == "header" else 0,
                "crc32": content_crc,
                "arrival": arrival,
            })
            self._camera_history = self._camera_history[-64:]

    def _camera_sample_is_new_locked(self, identity, content_crc):
        if identity == self._last_camera_key:
            return False
        if identity[0] == "header":
            sequence, stamp_ns = identity[1], identity[2]
            if (
                stamp_ns > 0
                and self._last_header_stamp_ns is not None
                and stamp_ns <= self._last_header_stamp_ns
            ):
                return False
            if (
                stamp_ns == 0
                and sequence > 0
                and self._last_header_seq is not None
                and sequence <= self._last_header_seq
            ):
                return False
        if content_crc == self._last_content_crc:
            return False
        return True

    def _lidar_callback(self, message):
        half_angle = math.radians(self.config.lidar_guard_half_angle_deg)
        values = []
        for index, distance in enumerate(message.ranges):
            raw_angle = (
                message.angle_min + index * message.angle_increment
                - math.radians(self.config.lidar_yaw_offset_deg)
            )
            angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))
            if abs(angle) > half_angle:
                continue
            if math.isinf(distance) and distance > 0.0:
                # Positive infinity is the standard clear/no-return reading.
                values.append(float("inf"))
                continue
            if not math.isfinite(distance):
                continue
            if distance < message.range_min or distance > message.range_max:
                continue
            values.append(float(distance))
        with self.lock:
            finite_values = [value for value in values if math.isfinite(value)]
            if finite_values:
                self.front_distance = min(finite_values)
            elif values:
                self.front_distance = None  # all +Inf means clear/no return
            else:
                self.front_distance = float("nan")  # no usable samples
            self.lidar_stamp = time.monotonic()

    def _semantic_callback(self, message):
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("semantic JSON root must be an object")
            unexpected = set(payload.keys()).difference(self.SEMANTIC_KEYS)
            if unexpected:
                raise ValueError(
                    "semantic message contains unsupported keys: %s"
                    % ",".join(sorted(str(key) for key in unexpected))
                )
        except (TypeError, ValueError) as exc:
            self._invalidate_semantic()
            self.rospy.logwarn_throttle(2.0, "Invalid semantic JSON: %s" % exc)
            return

        source_frame_seq = payload.get("source_frame_seq")
        source_stamp_ns = payload.get("source_stamp_ns")
        source_crc32 = payload.get("source_crc32")
        try:
            source_frame_seq = _semantic_source_int(
                source_frame_seq, "source_frame_seq"
            )
            source_stamp_ns = _semantic_source_int(
                source_stamp_ns, "source_stamp_ns"
            )
            source_crc32 = _semantic_source_int(source_crc32, "source_crc32")
            if not any((source_frame_seq, source_stamp_ns, source_crc32)):
                raise ValueError(
                    "semantic result needs a non-zero camera source identity"
                )
        except (TypeError, ValueError) as exc:
            self._invalidate_semantic()
            self.rospy.logwarn_throttle(2.0, "Invalid semantic source: %s" % exc)
            return

        now = time.monotonic()
        with self.lock:
            source_record = None
            for record in reversed(self._camera_history):
                if (
                    source_stamp_ns is not None
                    and source_stamp_ns > 0
                    and record["stamp_ns"] != source_stamp_ns
                ):
                    continue
                if (
                    source_frame_seq is not None
                    and source_frame_seq > 0
                    and record["header_seq"] != source_frame_seq
                ):
                    continue
                if (
                    source_crc32 is not None
                    and source_crc32 > 0
                    and record["crc32"] != source_crc32
                ):
                    continue
                source_record = record
                break
            if source_record is None:
                self._invalidate_semantic_locked()
                self.rospy.logwarn_throttle(
                    2.0, "Semantic result does not match a recent camera frame"
                )
                return
            if now - source_record["arrival"] > self.config.semantic_ttl_seconds:
                self._invalidate_semantic_locked()
                self.rospy.logwarn_throttle(2.0, "Semantic source frame is stale")
                return
            local_seq = source_record["local_seq"]
            if (
                self._last_semantic_source_local_seq is not None
                and local_seq <= self._last_semantic_source_local_seq
            ):
                self._invalidate_semantic_locked()
                self.rospy.logwarn_throttle(
                    2.0, "Semantic source frame is duplicate/out of order"
                )
                return
            try:
                observation = SemanticObservation(
                    sign_label=payload.get("sign_label"),
                    sign_confidence=payload.get("sign_confidence"),
                    signal_label=payload.get("signal_label"),
                    signal_confidence=payload.get("signal_confidence"),
                    latency_ms=payload.get("latency_ms"),
                    stamp=now,
                    source_frame_seq=source_frame_seq,
                    source_stamp_ns=source_stamp_ns,
                    source_local_seq=local_seq,
                    source_arrival_stamp=source_record["arrival"],
                )
            except (TypeError, ValueError) as exc:
                self._invalidate_semantic_locked()
                self.rospy.logwarn_throttle(2.0, "Invalid semantic JSON: %s" % exc)
                return
            self.semantic = observation
            self.semantic_stamp = now
            self.semantic_seq += 1
            self._last_semantic_source_local_seq = local_seq

    def _invalidate_semantic(self):
        with self.lock:
            self._invalidate_semantic_locked()

    def _invalidate_semantic_locked(self):
        """Revoke a previously accepted label after bad/stale AI output."""
        self.semantic = SemanticObservation()
        self.semantic_stamp = None
        self.semantic_seq += 1

    def _arm_service(self, unused_request):
        del unused_request
        now = time.monotonic()
        with self.lock:
            if self.estop_latched:
                return self._trigger_response_type(
                    success=False, message="E-stop is latched"
                )
            camera_ready = (
                self.frame_stamp is not None
                and now - self.frame_stamp <= self.config.camera_timeout_seconds
            )
            lidar_ready = (
                not self.use_lidar
                or (
                    self.lidar_stamp is not None
                    and now - self.lidar_stamp <= self.config.lidar_timeout_seconds
                )
            )
            if not camera_ready or not lidar_ready:
                return self._trigger_response_type(
                    success=False, message="camera/LiDAR is not fresh"
                )
            self.arm_requests += 1
            self.arm_request_stamp = now
        return self._trigger_response_type(
            success=True, message="one arm request queued"
        )

    def _estop_callback(self, message):
        if not bool(message.data):
            return
        with self.lock:
            self.estop_latched = True
        if self.emergency_stop_callback is not None:
            self.emergency_stop_callback()

    def snapshot(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return (
                frame,
                self.frame_stamp,
                self.frame_seq,
                self.front_distance,
                self.lidar_stamp,
                self.semantic,
                self.semantic_stamp,
                self.semantic_seq,
                self.arm_requests,
                self.arm_request_stamp,
                self.estop_latched,
            )


class Runtime(object):
    def __init__(self, config, scenario, actuator, telemetry, detector,
                 require_ai=False, stop_after_intersections=0):
        self.config = config
        self.perception = SmartCityPerception(config)
        self.provider = ScenarioDecisionProvider(scenario)
        self.fsm = SmartCityFSM(self.provider, config=config)
        self.actuator = actuator
        self.telemetry = telemetry
        self.detector = detector
        self.require_ai = require_ai
        self.stop_after_intersections = int(stop_after_intersections)
        if self.stop_after_intersections < 0:
            raise ValueError("stop_after_intersections must be non-negative")
        self.previous_lane_x = None
        self.last_perception = None
        self.last_semantic = SemanticObservation()
        self.last_seq = None
        self._last_external_semantic_seq = None
        self._last_external_source_local_seq = None
        self._last_external_fingerprint = None

    def arm(self, now):
        return self.fsm.arm(now=now)

    def step(self, frame, now, seq, camera_age=0.0, front_distance=None,
             external_semantic=None, semantic_seq=None):
        started = time.monotonic()
        new_camera_frame = self.last_perception is None or seq != self.last_seq
        if new_camera_frame:
            self.last_perception = self.perception.analyze(
                frame, previous_lane_x=self.previous_lane_x
            )
            if self.last_perception.lane_x_near is not None:
                self.previous_lane_x = self.last_perception.lane_x_near
            if external_semantic is None:
                self.last_semantic = self.detector.detect(frame)
            self.last_seq = seq
        if external_semantic is not None:
            # ROS semantics are asynchronous.  Accept a label only when it is
            # fresh, references a camera frame that already exists, and moves
            # both the semantic-result and camera-source sequences forward.
            self._update_external_semantic(
                external_semantic, semantic_seq, seq, now
            )

        semantic = self.last_semantic
        command = self.fsm.update(
            self.last_perception,
            now=now,
            frame_seq=seq,
            camera_age_seconds=camera_age,
            obstacle_distance_m=front_distance,
            ai_label=semantic.sign_label,
            ai_confidence=semantic.sign_confidence,
            signal_label=semantic.signal_label,
            signal_confidence=semantic.signal_confidence,
            ai_required=self.require_ai,
            semantic_seq=seq if semantic_seq is None else semantic_seq,
            semantic_source_frame_seq=semantic.source_local_seq,
        )
        # A camera-only hardware bench route must stop immediately after the
        # second intersection has been fully exited and its new lane is
        # stable.  Replace the positive intersection-cleared command before it
        # ever reaches the actuator, then latch the stop for the process.
        if (
            self.stop_after_intersections > 0
            and self.provider.consumed >= self.stop_after_intersections
            and command.reason == "intersection_cleared"
        ):
            self.fsm.emergency_stop("bench_route_complete", now=now)
            command = DriveCommand(
                0.0,
                0.0,
                SmartCityState.E_STOP,
                reason="bench_route_complete",
                action=command.action,
                intersection_id=command.intersection_id,
                timestamp=now,
            )
        if command.state == SmartCityState.E_STOP:
            self.actuator.emergency_stop()
        else:
            self.actuator.apply(command)
        loop_latency_ms = (time.monotonic() - started) * 1000.0
        self.telemetry.write(
            now,
            seq,
            command,
            self.last_perception,
            semantic,
            camera_age,
            front_distance,
            loop_latency_ms,
        )
        return command, draw_overlay(self.last_perception.debug_frame, command,
                                     semantic, loop_latency_ms)

    def _update_external_semantic(self, observation, semantic_seq,
                                  camera_seq, now):
        has_label = (
            observation.sign_label is not None
            or observation.signal_label is not None
        )
        if not has_label:
            self._clear_external_semantic(now, semantic_seq)
            return False

        if (
            isinstance(semantic_seq, bool)
            or not isinstance(semantic_seq, int)
            or semantic_seq <= 0
        ):
            self._clear_external_semantic(now)
            return False
        source_seq = observation.source_local_seq
        if (
            isinstance(source_seq, bool)
            or not isinstance(source_seq, int)
            or source_seq <= 0
            or source_seq > camera_seq
        ):
            self._clear_external_semantic(now, semantic_seq)
            return False

        result_age = now - observation.stamp
        source_arrival_stamp = observation.source_arrival_stamp
        if source_arrival_stamp is None:
            self._clear_external_semantic(now, semantic_seq)
            return False
        source_age = now - source_arrival_stamp
        if (
            result_age < -0.05
            or result_age > self.config.semantic_ttl_seconds
            or source_age < -0.05
            or source_age > self.config.semantic_ttl_seconds
        ):
            self._clear_external_semantic(now, semantic_seq)
            return False

        fingerprint = (
            observation.sign_label,
            observation.sign_confidence,
            observation.signal_label,
            observation.signal_confidence,
            observation.source_frame_seq,
            observation.source_stamp_ns,
            source_seq,
            source_arrival_stamp,
        )
        if self._last_external_semantic_seq is not None:
            if semantic_seq < self._last_external_semantic_seq:
                self._clear_external_semantic(now)
                return False
            if semantic_seq == self._last_external_semantic_seq:
                if fingerprint != self._last_external_fingerprint:
                    self._clear_external_semantic(now)
                    return False
                # Reusing the same accepted result between camera callbacks is
                # expected; the FSM sees the unchanged semantic_seq and does
                # not count it as another confirmation frame.
                self.last_semantic = observation
                return True
        if (
            self._last_external_source_local_seq is not None
            and source_seq <= self._last_external_source_local_seq
        ):
            self._clear_external_semantic(now, semantic_seq)
            return False

        self.last_semantic = observation
        self._last_external_semantic_seq = semantic_seq
        self._last_external_source_local_seq = source_seq
        self._last_external_fingerprint = fingerprint
        return True

    def _clear_external_semantic(self, now, semantic_seq=None):
        self.last_semantic = SemanticObservation(stamp=now)
        self._last_external_fingerprint = None
        if (
            isinstance(semantic_seq, int)
            and not isinstance(semantic_seq, bool)
            and semantic_seq > 0
            and (
                self._last_external_semantic_seq is None
                or semantic_seq > self._last_external_semantic_seq
            )
        ):
            self._last_external_semantic_seq = semantic_seq


def camera_message_identity(message):
    """Stable identity used to reject a driver replaying the same ROS frame."""
    header = getattr(message, "header", None)
    sequence = int(getattr(header, "seq", 0) or 0)
    stamp = getattr(header, "stamp", None)
    stamp_ns = 0
    if stamp is not None:
        if hasattr(stamp, "to_nsec"):
            stamp_ns = int(stamp.to_nsec())
        else:
            stamp_ns = (
                int(getattr(stamp, "secs", 0)) * 1000000000
                + int(getattr(stamp, "nsecs", 0))
            )
    if sequence != 0 or stamp_ns != 0:
        return ("header", sequence, stamp_ns)

    # Some camera bridges leave the ROS header at zero.  A checksum is a
    # conservative fallback: identical pixels are treated as a frozen frame.
    raw = memoryview(message.data)
    return (
        "sample",
        int(message.width),
        int(message.height),
        str(message.encoding),
        zlib.crc32(raw) & 0xFFFFFFFF,
    )


def camera_content_crc(message):
    """Checksum used to detect frozen pixels even when headers keep changing."""
    return zlib.crc32(memoryview(message.data)) & 0xFFFFFFFF


def ros_image_to_bgr(message):
    """Decode common sensor_msgs/Image encodings while respecting row stride."""
    encoding = str(message.encoding).lower()
    if encoding in ("bgr8", "rgb8"):
        channels = 3
    elif encoding in ("bgra8", "rgba8"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    else:
        raise ValueError("unsupported image encoding: %s" % message.encoding)

    raw = np.frombuffer(message.data, dtype=np.uint8)
    step = int(message.step) if int(message.step) > 0 else int(message.width) * channels
    expected = int(message.height) * step
    if raw.size < expected:
        raise ValueError("image buffer is shorter than height*step")
    rows = raw[:expected].reshape((int(message.height), step))
    pixels = rows[:, :int(message.width) * channels]
    if channels == 1:
        mono = pixels.reshape((int(message.height), int(message.width)))
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    image = pixels.reshape((int(message.height), int(message.width), channels))
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def draw_overlay(debug_frame, command, semantic, latency_ms):
    image = debug_frame.copy()
    colour = (0, 220, 0)
    if command.throttle <= 0.0:
        colour = (0, 190, 255)
    if command.state in (SmartCityState.SAFE_STOP, SmartCityState.E_STOP):
        colour = (0, 0, 255)
    lines = [
        "%s | %s" % (command.state, command.reason),
        "cmd steer=%+.2f throttle=%.2f action=%s" % (
            command.steering, command.throttle, command.action or "-"
        ),
        "AI sign=%s light=%s latency=%s ms" % (
            semantic.sign_label or "-",
            semantic.signal_label or "-",
            "-" if semantic.latency_ms is None else "%.1f" % semantic.latency_ms,
        ),
        "loop %.1f ms | q quit, e E-stop, x reset, a arm" % latency_ms,
        "manual: 1 RED, 2 YELLOW, 3 GREEN, c clear",
    ]
    for index, line in enumerate(lines):
        y = 24 + index * 24
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1, cv2.LINE_AA)
    return image


def handle_key(key, runtime, detector, now, allow_rearm=True):
    if key in (ord("q"), 27):
        return False
    if key == ord("e"):
        runtime.actuator.emergency_stop()
        runtime.fsm.emergency_stop("keyboard_estop", now=now)
    elif key == ord("x") and allow_rearm:
        runtime.fsm.reset_stop(now=now)
    elif key == ord("a") and allow_rearm:
        runtime.fsm.arm(now=now)
    elif key == ord("1") and hasattr(detector, "set_signal"):
        detector.set_signal("RED")
    elif key == ord("2") and hasattr(detector, "set_signal"):
        detector.set_signal("YELLOW")
    elif key == ord("3") and hasattr(detector, "set_signal"):
        detector.set_signal("GREEN")
    elif key == ord("c") and hasattr(detector, "clear"):
        detector.clear()
    elif key == ord("j") and hasattr(detector, "set_sign"):
        detector.set_sign("TURN_LEFT")
    elif key == ord("i") and hasattr(detector, "set_sign"):
        detector.set_sign("GO_STRAIGHT")
    elif key == ord("k") and hasattr(detector, "set_sign"):
        detector.set_sign("TURN_RIGHT")
    elif key == ord("u") and hasattr(detector, "set_sign"):
        detector.set_sign("NO_LEFT")
    elif key == ord("o") and hasattr(detector, "set_sign"):
        detector.set_sign("NO_RIGHT")
    return True


def run_offline(args, config, actuator, telemetry, detector):
    runtime = Runtime(config, args.scenario, actuator, telemetry, detector,
                      require_ai=args.require_ai,
                      stop_after_intersections=(
                          2 if args.bench_camera_only else 0
                      ))

    single_image = None
    capture = None
    simulated_clock = False
    if args.image:
        single_image = cv2.imread(args.image)
        if single_image is None:
            raise RuntimeError("Cannot read image: %s" % args.image)
    elif args.video:
        capture = cv2.VideoCapture(args.video)
        simulated_clock = True
    else:
        capture = cv2.VideoCapture(args.camera_index)
    if capture is not None and not capture.isOpened():
        raise RuntimeError("Cannot open camera/video source")

    fps = config.loop_hz
    if capture is not None and simulated_clock:
        measured_fps = capture.get(cv2.CAP_PROP_FPS)
        if measured_fps and measured_fps > 1.0:
            fps = measured_fps

    frame_index = 0
    armed = False
    try:
        while True:
            if single_image is not None:
                if frame_index > 0:
                    break
                frame = single_image
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            frame_index += 1
            now = frame_index / fps if simulated_clock else time.monotonic()
            if not armed:
                # Offline is always shadow mode, so auto-arm makes replay useful.
                runtime.arm(now)
                armed = True
            command, debug = runtime.step(frame, now, frame_index)
            if args.web_port:
                from src.debug.web_viewer import set_web_frame
                set_web_frame(debug)
            if args.display:
                cv2.imshow("Smart City V2 - SHADOW", debug)
                delay = 0 if single_image is not None else 1
                key = cv2.waitKey(delay) & 0xFF
                if not handle_key(key, runtime, detector, now):
                    break
            if args.max_frames and frame_index >= args.max_frames:
                break
            if command.state in (SmartCityState.FINISHED, SmartCityState.E_STOP):
                if not args.display:
                    break
    finally:
        actuator.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
    return 0


def run_ros(args, config, actuator, telemetry, detector):
    try:
        import rospy
    except ImportError:
        raise RuntimeError("ROS mode requires rospy/ROS Melodic")

    rospy.init_node("smart_city_v2", anonymous=False)
    if actuator is None:
        # Hardware constructors log through rospy, so initialise the node first.
        actuator = VehicleActuator(config)
    # Shutdown can race a control-loop apply().  Use the latched path so a
    # command already waiting on the actuator lock cannot re-energise motors.
    rospy.on_shutdown(actuator.emergency_stop)
    try:
        buffers = RosBuffers(
            rospy,
            config,
            use_lidar=args.use_lidar,
            semantic_topic=args.semantic_topic,
            emergency_stop_callback=actuator.emergency_stop,
        )
        runtime = Runtime(
            config,
            args.scenario,
            actuator,
            telemetry,
            detector,
            require_ai=args.require_ai,
            stop_after_intersections=(2 if args.bench_camera_only else 0),
        )
    except Exception:
        actuator.close()
        raise
    armed = False
    processed = 0
    last_debug = None
    bench_started_at = None
    semantic_worker = (
        LatestFrameSemanticWorker(detector)
        if getattr(args, "semantic_model", None) else None
    )
    last_semantic_submit_seq = None
    last_semantic_error_seq = None
    loop_period = 1.0 / max(1.0, config.loop_hz)

    rospy.loginfo("Smart City V2 started in %s mode" % (
        (
            "CAMERA BENCH MOTOR"
            if args.enable_motors and args.bench_camera_only
            else "MOTOR" if args.enable_motors else "SHADOW"
        )
    ))
    try:
        while not rospy.is_shutdown():
            now = time.monotonic()
            (frame, frame_stamp, seq, front_distance, lidar_stamp,
             topic_semantic, semantic_stamp, topic_semantic_seq, arm_requests,
             arm_request_stamp, estop_latched) = buffers.snapshot()

            if estop_latched:
                runtime.fsm.emergency_stop("ros_estop_latched", now)
                actuator.emergency_stop()

            if frame is None or frame_stamp is None:
                actuator.stop()
                time.sleep(loop_period)
                continue
            camera_age = max(0.0, now - frame_stamp)

            if args.use_lidar:
                if lidar_stamp is None:
                    actuator.stop()
                    time.sleep(loop_period)
                    continue
                if now - lidar_stamp > config.lidar_timeout_seconds:
                    runtime.fsm.emergency_stop("lidar_stale", now)
                    actuator.emergency_stop()

            if not armed:
                should_arm = should_arm_runtime(
                    args.enable_motors,
                    arm_requests,
                    arm_request_stamp,
                    now,
                    config.arm_request_ttl_seconds,
                    estop_latched,
                )
                if should_arm:
                    armed = runtime.arm(now)
                    if armed and args.bench_camera_only:
                        bench_started_at = now

            if (
                args.bench_camera_only
                and bench_started_at is not None
                and now - bench_started_at
                >= config.bench_max_runtime_seconds
            ):
                runtime.fsm.emergency_stop("bench_runtime_timeout", now)
                actuator.emergency_stop()
                rospy.logerr("Camera-only bench runtime limit reached; stopped")
                break

            semantic = None
            selected_semantic_seq = None
            if args.semantic_topic:
                if (
                    semantic_stamp is not None
                    and now - semantic_stamp <= config.semantic_ttl_seconds
                ):
                    semantic = topic_semantic
                else:
                    semantic = SemanticObservation()
                selected_semantic_seq = topic_semantic_seq
            elif semantic_worker is not None:
                if seq != last_semantic_submit_seq:
                    semantic_worker.submit(
                        frame,
                        source_local_seq=seq,
                        source_arrival_stamp=frame_stamp,
                    )
                    last_semantic_submit_seq = seq
                semantic, worker_seq, worker_error = semantic_worker.snapshot()
                # Reserve sequence 1 for the initial empty observation.  Every
                # completed inference, including failures, advances it.
                selected_semantic_seq = worker_seq + 1
                if (
                    worker_error is not None
                    and worker_seq != last_semantic_error_seq
                ):
                    rospy.logwarn("Semantic inference failed: %s" % worker_error)
                    last_semantic_error_seq = worker_seq

            command, last_debug = runtime.step(
                frame,
                now,
                seq,
                camera_age=camera_age,
                front_distance=front_distance if args.use_lidar else None,
                external_semantic=semantic,
                semantic_seq=(
                    selected_semantic_seq
                    if selected_semantic_seq is not None else seq
                ),
            )
            processed += 1

            if args.web_port:
                from src.debug.web_viewer import set_web_frame
                set_web_frame(last_debug)
            if args.display:
                cv2.imshow("Smart City V2", last_debug)
                key = cv2.waitKey(1) & 0xFF
                if not handle_key(
                    key,
                    runtime,
                    detector,
                    now,
                    allow_rearm=not args.enable_motors,
                ):
                    break
            if args.max_frames and processed >= args.max_frames:
                break
            if (
                args.bench_camera_only
                and command.reason == "bench_route_complete"
            ):
                rospy.loginfo(
                    "Camera bench LEFT-then-RIGHT route completed; motors latched off"
                )
                break
            time.sleep(loop_period)
    finally:
        if semantic_worker is not None:
            semantic_worker.close()
        actuator.close()
        cv2.destroyAllWindows()
    return 0


def load_config(path):
    config = SmartCityConfig()
    if path:
        with open(path, "r", encoding="utf-8") as config_file:
            values = json.load(config_file)
        if not isinstance(values, dict):
            raise ValueError("config JSON root must be an object")
        config.update(values)
    return config


def validate_live_inputs(args, config):
    """Refuse motor mode until route and calibration were explicitly signed."""
    if not (
        getattr(args, "semantic_topic", None)
        or getattr(args, "semantic_model", None)
    ):
        raise ValueError(
            "motor mode requires --semantic-topic or --semantic-model"
        )
    if getattr(args, "require_ai", False) is not True:
        raise ValueError("motor mode requires --require-ai")
    if (
        getattr(args, "mock_sign", None) is not None
        or getattr(args, "mock_signal", None) is not None
    ):
        raise ValueError("motor mode refuses mock sign/signal labels")
    if not args.config:
        raise ValueError("motor mode requires an explicit calibration JSON")
    config.validate_live()
    if os.path.abspath(args.scenario) == os.path.abspath(DEFAULT_SCENARIO):
        raise ValueError("motor mode refuses scenario_example.json")
    with open(args.scenario, "r", encoding="utf-8") as scenario_file:
        document = json.load(scenario_file)
    if not isinstance(document, dict):
        raise ValueError("live scenario root must be an object")
    if document.get("validated_for_live") is not True:
        raise ValueError("live scenario needs validated_for_live=true")
    route_id = document.get("route_id")
    if not isinstance(route_id, str) or not route_id.strip():
        raise ValueError("live scenario needs a non-empty route_id")
    intersections = document.get("intersections")
    if not isinstance(intersections, list):
        raise ValueError("live scenario intersections must be an array")
    for index, entry in enumerate(intersections):
        if (
            isinstance(entry, dict)
            and "requires_sign" in entry
            and not isinstance(entry.get("requires_sign"), bool)
        ):
            raise ValueError(
                "live scenario entry %d requires_sign must be boolean"
                % (index + 1)
            )
        if isinstance(entry, dict) and (
            "mock_sign" in entry or "mock_signal" in entry
        ):
            raise ValueError(
                "live scenario entry %d contains a mock label" % (index + 1)
            )
    if not args.use_lidar:
        raise ValueError("motor mode requires --use-lidar")


def validate_camera_bench_inputs(args, config):
    """Validate the deliberately narrow camera-only motor bench.

    This is not a shortcut into competition live mode. It accepts exactly two
    scripted intersections (LEFT then RIGHT), no AI/mock/LiDAR inputs, and a
    bench-only configuration with its own hard motion envelope.
    """
    if not args.config:
        raise ValueError("camera bench requires an explicit config JSON")
    config.validate_camera_bench()
    if os.path.abspath(args.scenario) == os.path.abspath(DEFAULT_SCENARIO):
        raise ValueError("camera bench refuses scenario_example.json")
    if getattr(args, "use_lidar", False):
        raise ValueError("camera bench is camera-only; remove --use-lidar")
    if getattr(args, "semantic_topic", None):
        raise ValueError("camera bench refuses --semantic-topic")
    if getattr(args, "semantic_model", None):
        raise ValueError("camera bench refuses --semantic-model")
    if getattr(args, "require_ai", False):
        raise ValueError("camera bench refuses --require-ai")
    if (
        getattr(args, "mock_sign", None) is not None
        or getattr(args, "mock_signal", None) is not None
    ):
        raise ValueError("camera bench refuses mock sign/signal labels")

    with open(args.scenario, "r", encoding="utf-8") as scenario_file:
        document = json.load(scenario_file)
    if not isinstance(document, dict):
        raise ValueError("camera bench scenario root must be an object")
    if document.get("bench_only") is not True:
        raise ValueError("camera bench scenario needs bench_only=true")
    if document.get("validated_for_live") is not False:
        raise ValueError(
            "camera bench scenario must keep validated_for_live=false"
        )
    route_id = document.get("route_id")
    if not isinstance(route_id, str) or not route_id.strip():
        raise ValueError("camera bench scenario needs a non-empty route_id")

    intersections = document.get("intersections")
    if not isinstance(intersections, list) or len(intersections) != 2:
        raise ValueError("camera bench needs exactly two intersections")
    expected_actions = ("LEFT", "RIGHT")
    for index, expected in enumerate(expected_actions):
        entry = intersections[index]
        if not isinstance(entry, dict):
            raise ValueError("camera bench intersection must be an object")
        intersection_id = entry.get("id")
        if not isinstance(intersection_id, str) or not intersection_id.strip():
            raise ValueError("camera bench intersection needs a non-empty id")
        action = entry.get("action")
        if not isinstance(action, str) or action.strip().upper() != expected:
            raise ValueError(
                "camera bench action %d must be %s" % (index + 1, expected)
            )
        allowed = entry.get("allowed")
        if not isinstance(allowed, list):
            raise ValueError("camera bench allowed exits must be an array")
        normalised_allowed = []
        for value in allowed:
            if not isinstance(value, str):
                raise ValueError("camera bench allowed exit must be a string")
            direction = value.strip().upper()
            if direction not in ("LEFT", "STRAIGHT", "RIGHT"):
                raise ValueError("camera bench contains an invalid exit")
            normalised_allowed.append(direction)
        if expected not in normalised_allowed:
            raise ValueError("camera bench action is not an allowed exit")
        if any(key in entry for key in (
            "mock_sign", "mock_signal", "requires_sign",
        )):
            raise ValueError("camera bench intersections must be scripted only")


def should_arm_runtime(enable_motors, arm_requests, arm_request_stamp,
                       now, request_ttl, estop_latched):
    """Return whether this tick may arm without a startup auto-arm race.

    Shadow mode starts automatically.  Motor mode requires a fresh one-shot
    service request made after camera/LiDAR became ready; the ``--arm`` CLI flag
    only acknowledges that policy and never moves the car by itself.
    """
    if not enable_motors:
        return True
    return bool(
        arm_requests > 0
        and arm_request_stamp is not None
        and 0.0 <= now - arm_request_stamp <= request_ttl
        and not estop_latched
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Smart City V2 deterministic controller (motors off by default)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ros", action="store_true", help="read ROS camera topics")
    source.add_argument("--video", help="offline video replay")
    source.add_argument("--image", help="inspect one image")
    source.add_argument("--camera-index", type=int, help="OpenCV camera index")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--config", help="JSON overrides for SmartCityConfig")
    parser.add_argument("--log", help="optional CSV telemetry output")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--web-port", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument(
        "--bench-camera-only",
        action="store_true",
        help=(
            "supervised low-speed LEFT-then-RIGHT hardware bench; no AI or "
            "LiDAR and never valid for competition"
        ),
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help=(
            "enable the one-shot ROS arm service; this acknowledgement is "
            "required with motors but never auto-arms the car"
        ),
    )
    parser.add_argument("--use-lidar", action="store_true")

    semantic_source = parser.add_mutually_exclusive_group()
    semantic_source.add_argument(
        "--semantic-topic",
        help="ROS std_msgs/String JSON produced by an external AI node",
    )
    semantic_source.add_argument(
        "--semantic-model",
        help="local YOLO detect checkpoint for signs/lights",
    )
    parser.add_argument(
        "--semantic-device",
        help="Ultralytics device, for example 0 or cpu (default: auto)",
    )
    parser.add_argument(
        "--semantic-imgsz", type=int, default=640,
        help="YOLO inference size (default: 640)",
    )
    parser.add_argument("--require-ai", action="store_true",
                        help="hold at junction until a fresh traffic-light result exists")
    parser.add_argument("--mock-sign")
    parser.add_argument("--mock-signal")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.bench_camera_only and not args.ros:
        raise SystemExit("--bench-camera-only is allowed only with --ros")
    if args.enable_motors and not args.ros:
        raise SystemExit("--enable-motors is allowed only with --ros")
    if args.enable_motors and not args.arm:
        raise SystemExit("live motion requires both --enable-motors and --arm")
    if args.semantic_topic and not args.ros:
        raise SystemExit("--semantic-topic requires --ros")
    if args.require_ai and not (
        args.semantic_topic or args.semantic_model
        or args.mock_sign or args.mock_signal
    ):
        raise SystemExit(
            "--require-ai needs --semantic-topic, --semantic-model or a mock label"
        )
    if args.semantic_model and (args.mock_sign or args.mock_signal):
        raise SystemExit("--semantic-model cannot be combined with mock labels")

    config = load_config(args.config)
    if args.bench_camera_only:
        try:
            validate_camera_bench_inputs(args, config)
        except (IOError, OSError, TypeError, ValueError) as exc:
            raise SystemExit("camera bench safety check failed: %s" % exc)
    elif args.enable_motors:
        try:
            validate_live_inputs(args, config)
        except (IOError, OSError, TypeError, ValueError) as exc:
            raise SystemExit("live safety check failed: %s" % exc)
    telemetry = CsvTelemetry(args.log)
    if args.semantic_model:
        try:
            detector = YoloSemanticDetector(
                model_path=args.semantic_model,
                min_confidence=config.ai_min_confidence,
                image_size=args.semantic_imgsz,
                device=args.semantic_device,
            )
        except (IOError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise SystemExit("cannot load semantic model: %s" % exc)
    else:
        detector = ManualSemanticDetector(
            sign_label=args.mock_sign,
            signal_label=args.mock_signal,
        )
    # ROS hardware is constructed inside run_ros, after rospy.init_node().
    actuator = None if args.enable_motors else NullActuator()

    web_server = None
    if args.web_port:
        from src.debug.web_viewer import start_web_stream_server
        web_server = start_web_stream_server(args.web_port)
    try:
        if args.ros:
            return run_ros(args, config, actuator, telemetry, detector)
        return run_offline(args, config, actuator, telemetry, detector)
    finally:
        if actuator is not None:
            actuator.close()
        telemetry.close()
        if web_server is not None:
            web_server.shutdown()


def _optional_number(value):
    if value is None:
        return ""
    return "%.5f" % float(value)


def _semantic_source_int(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % name)
    if value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


if __name__ == "__main__":
    sys.exit(main())
