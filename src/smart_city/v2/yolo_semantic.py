# -*- coding: utf-8 -*-
"""YOLO traffic-sign/light adapter for the deterministic Smart City V2 FSM.

The teammate model contains scene-geometry classes as well as semantic
classes.  This module deliberately ignores ``Corner``, ``Decision`` and
``Interact``.  It returns only :class:`SemanticObservation` values and never
returns steering, throttle, exits or crosswalk geometry.

Ultralytics is imported lazily so the rest of Smart City V2 and its unit tests
continue to run on machines which do not have the inference dependency.
"""

from __future__ import absolute_import

import math
import os
import threading
import time

from .semantic import SemanticObservation


EXPECTED_CLASS_NAMES = frozenset((
    "CORNER",
    "DECISION",
    "FORBIDDEN",
    "GREEN_LIGHT",
    "INTERACT",
    "LEFT",
    "RED_LIGHT",
    "RIGHT",
    "STRAIGHT",
))

IGNORED_GEOMETRY_CLASSES = frozenset(("CORNER", "DECISION", "INTERACT"))

_DIRECT_SIGN_LABELS = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "STRAIGHT": "STRAIGHT",
}


def _normalise_class_name(value):
    if not isinstance(value, str):
        return None
    value = value.strip().upper().replace("-", "_").replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or None


def _finite_ratio(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("%s must be in [0, 1]" % name)
    return value


def forbidden_bbox_to_label(
        xyxy,
        frame_width,
        left_max_ratio=0.38,
        center_min_ratio=0.43,
        center_max_ratio=0.57,
        right_min_ratio=0.62):
    """Convert a generic Forbidden box into a direction prohibition.

    The official setup described by the team places the sign on the left,
    centre or right side of the face looking at the car.  Small dead bands
    between those zones are intentional: an ambiguous position returns None
    and the sign-required FSM holds instead of guessing a turn.
    """

    if frame_width is None or float(frame_width) <= 0.0:
        raise ValueError("frame_width must be positive")
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        raise ValueError("xyxy must contain four coordinates")
    left_max_ratio = _finite_ratio(left_max_ratio, "left_max_ratio")
    center_min_ratio = _finite_ratio(center_min_ratio, "center_min_ratio")
    center_max_ratio = _finite_ratio(center_max_ratio, "center_max_ratio")
    right_min_ratio = _finite_ratio(right_min_ratio, "right_min_ratio")
    if not (
        left_max_ratio < center_min_ratio
        <= center_max_ratio < right_min_ratio
    ):
        raise ValueError("forbidden sign zones must be ordered and disjoint")

    x1, unused_y1, x2, unused_y2 = [float(value) for value in xyxy]
    del unused_y1, unused_y2
    if x2 < x1:
        raise ValueError("xyxy has a negative width")
    centre_ratio = ((x1 + x2) * 0.5) / float(frame_width)
    if centre_ratio < 0.0 or centre_ratio > 1.0:
        return None
    if centre_ratio <= left_max_ratio:
        return "NO_LEFT"
    if center_min_ratio <= centre_ratio <= center_max_ratio:
        return "NO_STRAIGHT"
    if centre_ratio >= right_min_ratio:
        return "NO_RIGHT"
    return None


def resolve_semantic_detections(
        detections,
        frame_width,
        min_confidence=0.60,
        conflict_margin=0.12,
        forbidden_zones=None):
    """Reduce YOLO detections to one traffic sign and one light state.

    Red always wins a simultaneous red/green conflict.  Conflicting sign
    classes must have a clear confidence winner; otherwise no sign is emitted.
    Each detection is a dict with ``class_name``, ``confidence`` and ``xyxy``.
    """

    min_confidence = _finite_ratio(min_confidence, "min_confidence")
    conflict_margin = _finite_ratio(conflict_margin, "conflict_margin")
    zones = forbidden_zones or {}
    signs = {}
    red_confidence = None
    green_confidence = None

    for detection in detections:
        if not isinstance(detection, dict):
            continue
        class_name = _normalise_class_name(detection.get("class_name"))
        try:
            confidence = float(detection.get("confidence"))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(confidence)
            or confidence < min_confidence
            or confidence > 1.0
        ):
            continue
        if class_name in IGNORED_GEOMETRY_CLASSES:
            continue
        if class_name == "RED_LIGHT":
            red_confidence = max(red_confidence or 0.0, confidence)
            continue
        if class_name == "GREEN_LIGHT":
            green_confidence = max(green_confidence or 0.0, confidence)
            continue

        sign_label = _DIRECT_SIGN_LABELS.get(class_name)
        if class_name == "FORBIDDEN":
            try:
                sign_label = forbidden_bbox_to_label(
                    detection.get("xyxy"), frame_width, **zones
                )
            except (TypeError, ValueError):
                sign_label = None
        if sign_label is not None:
            signs[sign_label] = max(signs.get(sign_label, 0.0), confidence)

    signal_label = None
    signal_confidence = None
    if red_confidence is not None:
        # Fail-safe policy: never allow GREEN to mask a concurrent RED box.
        signal_label = "RED"
        signal_confidence = red_confidence
    elif green_confidence is not None:
        signal_label = "GREEN"
        signal_confidence = green_confidence

    sign_label = None
    sign_confidence = None
    ranked_signs = sorted(
        signs.items(), key=lambda item: (-item[1], item[0])
    )
    if ranked_signs:
        if (
            len(ranked_signs) == 1
            or ranked_signs[0][1] - ranked_signs[1][1] >= conflict_margin
        ):
            sign_label, sign_confidence = ranked_signs[0]

    return SemanticObservation(
        sign_label=sign_label,
        sign_confidence=sign_confidence,
        signal_label=signal_label,
        signal_confidence=signal_confidence,
    )


def _to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _class_name(names, class_index):
    if isinstance(names, dict):
        return names.get(class_index, names.get(str(class_index)))
    try:
        return names[class_index]
    except (IndexError, KeyError, TypeError):
        return None


class YoloSemanticDetector(object):
    """Synchronous Ultralytics detector; use the worker in ROS mode."""

    def __init__(self, model_path=None, min_confidence=0.60,
                 image_size=640, device=None, conflict_margin=0.12,
                 forbidden_zones=None, model=None):
        self.model_path = model_path
        self.min_confidence = _finite_ratio(
            min_confidence, "min_confidence"
        )
        self.conflict_margin = _finite_ratio(
            conflict_margin, "conflict_margin"
        )
        self.image_size = int(image_size)
        if self.image_size < 160:
            raise ValueError("image_size must be at least 160")
        self.device = device
        self.forbidden_zones = dict(forbidden_zones or {})

        if model is None:
            if not isinstance(model_path, str) or not model_path.strip():
                raise ValueError("semantic model path is required")
            if not os.path.isfile(model_path):
                raise IOError("semantic model does not exist: %s" % model_path)
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "semantic YOLO needs the ultralytics package"
                ) from exc
            model = YOLO(model_path, task="detect")
        self.model = model
        self._validate_model_names(getattr(self.model, "names", None))

    @staticmethod
    def _validate_model_names(names):
        if isinstance(names, dict):
            values = list(names.values())
        elif isinstance(names, (list, tuple)):
            values = list(names)
        else:
            raise ValueError("semantic model has no readable class names")
        normalised = frozenset(
            name for name in (_normalise_class_name(value) for value in values)
            if name is not None
        )
        missing = EXPECTED_CLASS_NAMES.difference(normalised)
        if missing:
            raise ValueError(
                "semantic model is missing classes: %s"
                % ",".join(sorted(missing))
            )

    def detect(self, frame):
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("semantic detector needs a BGR image")
        started = time.monotonic()
        kwargs = {
            "source": frame,
            "imgsz": self.image_size,
            "conf": self.min_confidence,
            "verbose": False,
        }
        if self.device is not None and str(self.device).strip():
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        detections = []
        for result in results or ():
            names = getattr(result, "names", None) or getattr(
                self.model, "names", None
            )
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            classes = _to_list(getattr(boxes, "cls", ()))
            confidences = _to_list(getattr(boxes, "conf", ()))
            coordinates = _to_list(getattr(boxes, "xyxy", ()))
            for raw_class, confidence, xyxy in zip(
                    classes, confidences, coordinates):
                class_index = int(raw_class)
                detections.append({
                    "class_name": _class_name(names, class_index),
                    "confidence": float(confidence),
                    "xyxy": list(xyxy),
                })

        observation = resolve_semantic_detections(
            detections,
            frame_width=int(frame.shape[1]),
            min_confidence=self.min_confidence,
            conflict_margin=self.conflict_margin,
            forbidden_zones=self.forbidden_zones,
        )
        observation.latency_ms = (time.monotonic() - started) * 1000.0
        return observation


class LatestFrameSemanticWorker(object):
    """Single-worker, latest-frame-only inference for the ROS control loop.

    The camera callback/control loop never waits for YOLO.  A newer submitted
    frame replaces a pending older frame.  Completed results retain the source
    frame identity and arrival timestamp so Runtime can reject stale output.
    """

    def __init__(self, detector):
        if not hasattr(detector, "detect"):
            raise TypeError("detector must provide detect(frame)")
        self.detector = detector
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._result = SemanticObservation()
        self._result_seq = 0
        self._last_error = None
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def submit(self, frame, source_local_seq, source_arrival_stamp,
               source_frame_seq=None, source_stamp_ns=None):
        if frame is None:
            raise ValueError("frame is required")
        if (
            isinstance(source_local_seq, bool)
            or not isinstance(source_local_seq, int)
            or source_local_seq <= 0
        ):
            raise ValueError("source_local_seq must be positive")
        source_arrival_stamp = float(source_arrival_stamp)
        if (
            not math.isfinite(source_arrival_stamp)
            or source_arrival_stamp < 0.0
        ):
            raise ValueError(
                "source_arrival_stamp must be finite and non-negative"
            )
        for name, value in (
                ("source_frame_seq", source_frame_seq),
                ("source_stamp_ns", source_stamp_ns)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError("%s must be a non-negative integer" % name)
        submitted = (
            frame.copy(),
            source_local_seq,
            source_arrival_stamp,
            source_frame_seq,
            source_stamp_ns,
        )
        with self._condition:
            if self._closed:
                return False
            self._pending = submitted
            self._condition.notify()
        return True

    def snapshot(self):
        with self._condition:
            return self._result, self._result_seq, self._last_error

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _loop(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait(timeout=0.25)
                if self._closed:
                    return
                (frame, local_seq, arrival_stamp,
                 frame_seq, stamp_ns) = self._pending
                self._pending = None
            error = None
            try:
                detected = self.detector.detect(frame)
                if not isinstance(detected, SemanticObservation):
                    raise TypeError("detector must return SemanticObservation")
                result = SemanticObservation(
                    sign_label=detected.sign_label,
                    sign_confidence=detected.sign_confidence,
                    signal_label=detected.signal_label,
                    signal_confidence=detected.signal_confidence,
                    latency_ms=detected.latency_ms,
                    stamp=time.monotonic(),
                    source_frame_seq=frame_seq,
                    source_stamp_ns=stamp_ns,
                    source_local_seq=local_seq,
                    source_arrival_stamp=arrival_stamp,
                )
            except Exception as exc:
                # Publish an empty result for this source so an earlier GREEN
                # can never survive an inference failure.
                error = "%s: %s" % (exc.__class__.__name__, exc)
                result = SemanticObservation(
                    stamp=time.monotonic(),
                    source_frame_seq=frame_seq,
                    source_stamp_ns=stamp_ns,
                    source_local_seq=local_seq,
                    source_arrival_stamp=arrival_stamp,
                )
            with self._condition:
                self._result = result
                self._result_seq += 1
                self._last_error = error


__all__ = (
    "EXPECTED_CLASS_NAMES",
    "IGNORED_GEOMETRY_CLASSES",
    "LatestFrameSemanticWorker",
    "YoloSemanticDetector",
    "forbidden_bbox_to_label",
    "resolve_semantic_detections",
)
