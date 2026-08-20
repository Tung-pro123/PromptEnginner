# -*- coding: utf-8 -*-
"""Narrow contract between a future AI detector and the deterministic FSM.

The detector is allowed to describe a sign/light only.  It cannot return
steering or throttle, which keeps model mistakes out of the actuator layer.
"""

from __future__ import absolute_import

import math
import threading
import time


class SemanticObservation(object):
    __slots__ = (
        "sign_label",
        "sign_confidence",
        "signal_label",
        "signal_confidence",
        "crosswalk_conf",
        "left_conf",
        "right_conf",
        "latency_ms",
        "stamp",
        "source_frame_seq",
        "source_stamp_ns",
        "source_local_seq",
    )

    def __init__(
        self,
        sign_label=None,
        sign_confidence=None,
        signal_label=None,
        signal_confidence=None,
        crosswalk_conf=None,
        left_conf=None,
        right_conf=None,
        latency_ms=None,
        stamp=None,
        source_frame_seq=None,
        source_stamp_ns=None,
        source_local_seq=None,
    ):
        self.sign_label = _normalise(sign_label)
        self.sign_confidence = _confidence(sign_confidence)
        self.signal_label = _normalise(signal_label)
        self.signal_confidence = _confidence(signal_confidence)
        self.crosswalk_conf = _confidence(crosswalk_conf)
        self.left_conf = _confidence(left_conf)
        self.right_conf = _confidence(right_conf)
        self.latency_ms = None if latency_ms is None else float(latency_ms)
        self.stamp = time.monotonic() if stamp is None else float(stamp)
        self.source_frame_seq = _optional_non_negative_int(
            source_frame_seq, "source_frame_seq"
        )
        self.source_stamp_ns = _optional_non_negative_int(
            source_stamp_ns, "source_stamp_ns"
        )
        self.source_local_seq = _optional_non_negative_int(
            source_local_seq, "source_local_seq"
        )

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


class NullSemanticDetector(object):
    """Used before the AI model exists; scenario.json supplies decisions."""

    def detect(self, frame):
        del frame
        return SemanticObservation()


class ManualSemanticDetector(object):
    """Thread-safe labels for dry-run/bench tests without an AI model."""

    def __init__(self, sign_label=None, signal_label=None, confidence=1.0):
        self._lock = threading.Lock()
        self._sign_label = sign_label
        self._signal_label = signal_label
        self._sign_confidence = confidence
        self._signal_confidence = confidence

    def set_sign(self, label, confidence=1.0):
        with self._lock:
            self._sign_label = label
            self._sign_confidence = confidence

    def set_signal(self, label, confidence=1.0):
        with self._lock:
            self._signal_label = label
            self._signal_confidence = confidence

    def clear(self):
        with self._lock:
            self._sign_label = None
            self._signal_label = None

    def detect(self, frame):
        del frame
        started = time.monotonic()
        with self._lock:
            sign = self._sign_label
            signal = self._signal_label
            sign_confidence = self._sign_confidence
            signal_confidence = self._signal_confidence
        return SemanticObservation(
            sign_label=sign,
            sign_confidence=sign_confidence if sign is not None else None,
            signal_label=signal,
            signal_confidence=signal_confidence if signal is not None else None,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )


class FunctionSemanticDetector(object):
    """Adapter for the teammate's model callable: ``model(frame) -> dict``.

    Accepted keys are only ``sign_label``, ``sign_confidence``,
    ``signal_label`` and ``signal_confidence``.  Control-like outputs are
    rejected explicitly.
    """

    _CONTROL_KEYS = frozenset((
        "steering", "throttle", "speed", "motor", "servo", "action"
    ))

    def __init__(self, predict_function):
        if not callable(predict_function):
            raise TypeError("predict_function must be callable")
        self.predict_function = predict_function

    def detect(self, frame):
        started = time.monotonic()
        output = self.predict_function(frame)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if isinstance(output, SemanticObservation):
            output.latency_ms = elapsed_ms
            return output
        if not isinstance(output, dict):
            raise TypeError("AI output must be SemanticObservation or dict")
        normalised_keys = {
            key.lower() for key in output.keys() if isinstance(key, str)
        }
        forbidden = self._CONTROL_KEYS.intersection(normalised_keys)
        if forbidden:
            raise ValueError(
                "AI may not output actuator/control keys: %s"
                % ",".join(sorted(forbidden))
            )
        return SemanticObservation(
            sign_label=output.get("sign_label"),
            sign_confidence=output.get("sign_confidence"),
            signal_label=output.get("signal_label"),
            signal_confidence=output.get("signal_confidence"),
            crosswalk_conf=output.get("crosswalk_conf"),
            left_conf=output.get("left_conf"),
            right_conf=output.get("right_conf"),
            latency_ms=elapsed_ms,
        )


def _normalise(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("semantic label must be a string")
    value = value.strip().upper().replace("-", "_").replace(" ", "_")
    return value or None


def _confidence(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return value


def _optional_non_negative_int(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % name)
    if value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


__all__ = (
    "FunctionSemanticDetector",
    "ManualSemanticDetector",
    "NullSemanticDetector",
    "SemanticObservation",
)
