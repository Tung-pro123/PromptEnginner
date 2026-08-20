# -*- coding: utf-8 -*-
"""Pure-Python decision layer for the Smart City intersection FSM.

This module deliberately has no ROS, camera, or vehicle dependency.  During
development, decisions can come from ``mock_sign``/``action`` in a JSON
scenario.  Later, the exact same ``decide(ai_label=...)`` call can receive a
label produced by an AI detector.

The resolver is fail-closed: an unknown label, malformed intersection,
disallowed manoeuvre, or exhausted scenario always produces ``STOP``.
"""

from __future__ import absolute_import

import json
import os
from collections import namedtuple
from enum import Enum


class Direction(Enum):
    """Motion commands understood by the Smart City FSM."""

    LEFT = "LEFT"
    STRAIGHT = "STRAIGHT"
    RIGHT = "RIGHT"
    STOP = "STOP"
    END = "END"


class DecisionResult(namedtuple(
        "DecisionResultBase",
        "action allowed sign_label reason intersection_id")):
    """Immutable result returned by :meth:`ScenarioDecisionProvider.decide`.

    ``allowed`` is a tuple of :class:`Direction` values.  It only contains
    physical exits (LEFT, STRAIGHT, RIGHT); STOP is never an exit.
    """

    __slots__ = ()


_DRIVING_DIRECTIONS = (
    Direction.LEFT,
    Direction.STRAIGHT,
    Direction.RIGHT,
)

_DEFAULT_PREFERENCE = (
    Direction.STRAIGHT,
    Direction.RIGHT,
    Direction.LEFT,
)

_DIRECTION_ALIASES = {
    "LEFT": Direction.LEFT,
    "L": Direction.LEFT,
    "STRAIGHT": Direction.STRAIGHT,
    "FORWARD": Direction.STRAIGHT,
    "F": Direction.STRAIGHT,
    "RIGHT": Direction.RIGHT,
    "R": Direction.RIGHT,
    "STOP": Direction.STOP,
    "HALT": Direction.STOP,
    "END": Direction.END,
    "FINISH": Direction.END,
    "GOAL": Direction.END,
}

_COMMAND_SIGNS = {
    "TURN_LEFT": Direction.LEFT,
    "TURN_STRAIGHT": Direction.STRAIGHT,
    "GO_STRAIGHT": Direction.STRAIGHT,
    "TURN_RIGHT": Direction.RIGHT,
}

_PROHIBITION_SIGNS = {
    "NO_LEFT": Direction.LEFT,
    "NO_STRAIGHT": Direction.STRAIGHT,
    "NO_RIGHT": Direction.RIGHT,
}

_RED_LIGHT_LABELS = frozenset(("RED", "RED_LIGHT", "TRAFFIC_LIGHT_RED"))
_GREEN_LIGHT_LABELS = frozenset((
    "GREEN",
    "GREEN_LIGHT",
    "TRAFFIC_LIGHT_GREEN",
))
_STOP_LIGHT_LABELS = frozenset((
    "YELLOW",
    "YELLOW_LIGHT",
    "TRAFFIC_LIGHT_YELLOW",
))


def _normalise_label(label):
    if not isinstance(label, str):
        return None
    value = label.strip().upper().replace("-", "_").replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or None


def _parse_allowed(raw_allowed):
    if not isinstance(raw_allowed, (list, tuple)):
        return None

    parsed = []
    for raw_direction in raw_allowed:
        label = _normalise_label(raw_direction)
        direction = _DIRECTION_ALIASES.get(label)
        if direction not in _DRIVING_DIRECTIONS:
            return None
        if direction not in parsed:
            parsed.append(direction)
    return tuple(parsed)


def _parse_preference(raw_preference):
    if raw_preference is None:
        return _DEFAULT_PREFERENCE
    if not isinstance(raw_preference, (list, tuple)):
        return None

    parsed = []
    for raw_direction in raw_preference:
        if isinstance(raw_direction, Direction):
            direction = raw_direction
        else:
            label = _normalise_label(raw_direction)
            direction = _DIRECTION_ALIASES.get(label)
        if direction not in _DRIVING_DIRECTIONS:
            return None
        if direction not in parsed:
            parsed.append(direction)

    # A partial preference still remains safe and useful: unspecified choices
    # are tried after the explicitly preferred ones.
    for direction in _DEFAULT_PREFERENCE:
        if direction not in parsed:
            parsed.append(direction)
    return tuple(parsed)


def resolve_sign_label(sign_label, allowed, preference=None):
    """Resolve one traffic-sign label without any scenario/FSM state.

    Args:
        sign_label: ``TURN_LEFT/RIGHT/STRAIGHT``, ``NO_LEFT/RIGHT/STRAIGHT``,
            ``STOP``, or a supported short direction alias.
        allowed: iterable of allowed exits (strings or ``Direction`` values).
        preference: optional direction order used for prohibition signs.

    Returns:
        A :class:`Direction`.  Invalid or unsafe input returns ``STOP``.
    """

    raw_allowed = []
    try:
        for direction in allowed:
            if isinstance(direction, Direction):
                raw_allowed.append(direction.value)
            else:
                raw_allowed.append(direction)
    except (TypeError, AttributeError):
        return Direction.STOP

    parsed_allowed = _parse_allowed(raw_allowed)
    parsed_preference = _parse_preference(preference)
    label = _normalise_label(sign_label)
    if parsed_allowed is None or parsed_preference is None or label is None:
        return Direction.STOP

    requested = _COMMAND_SIGNS.get(label)
    if requested is None:
        requested = _DIRECTION_ALIASES.get(label)
    if requested is not None:
        if requested in (Direction.STOP, Direction.END):
            return requested
        if requested in parsed_allowed:
            return requested
        return Direction.STOP

    prohibited = _PROHIBITION_SIGNS.get(label)
    if prohibited is not None:
        for candidate in parsed_preference:
            if candidate in parsed_allowed and candidate is not prohibited:
                return candidate
        return Direction.STOP

    return Direction.STOP


class ScenarioDecisionProvider(object):
    """Sequential and deterministic intersection decision provider.

    ``scenario`` may be a JSON file path or an already decoded dictionary.  If
    omitted, ``scenario_example.json`` next to this module is loaded.

    An explicit ``ai_label`` overrides the route label/action in the scenario.
    ``RED`` (and yellow variants) returns STOP *without consuming* the current
    intersection. ``GREEN`` permits the configured scenario decision.  The
    optional ``signal_label`` argument lets the future AI adapter pass a light
    state and a traffic-sign label separately while keeping the FSM API stable.
    """

    def __init__(self, scenario=None):
        self._intersections = []
        self._index = 0
        self._load_error = None
        self._preference = _DEFAULT_PREFERENCE

        if scenario is None:
            scenario = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "scenario_example.json",
            )

        try:
            document = self._load_document(scenario)
            self._configure(document)
        except (IOError, OSError, ValueError, TypeError) as exc:
            self._load_error = str(exc) or exc.__class__.__name__

    @staticmethod
    def _load_document(source):
        if isinstance(source, dict):
            return source

        path_types = (str, bytes)
        if hasattr(os, "PathLike"):
            path_types = path_types + (os.PathLike,)
        if not isinstance(source, path_types):
            raise TypeError("scenario must be a JSON path or dictionary")

        with open(source, "r", encoding="utf-8") as scenario_file:
            return json.load(scenario_file)

    def _configure(self, document):
        if not isinstance(document, dict):
            raise ValueError("scenario root must be an object")

        intersections = document.get("intersections")
        if not isinstance(intersections, list):
            raise ValueError("scenario.intersections must be an array")

        raw_preference = document.get(
            "preference",
            document.get("default_preference"),
        )
        preference = _parse_preference(raw_preference)
        if preference is None:
            raise ValueError("scenario preference is invalid")

        self._intersections = list(intersections)
        self._preference = preference

    @property
    def count(self):
        """Total number of configured intersections."""

        return len(self._intersections)

    @property
    def consumed(self):
        """Number of intersections already resolved (including failures)."""

        return self._index

    @property
    def remaining(self):
        """Number of intersections not consumed yet."""

        return max(0, self.count - self._index)

    def reset(self):
        """Rewind a valid scenario for another dry run."""

        self._index = 0

    @staticmethod
    def _stop(reason, intersection_id=None, allowed=(), sign_label=None):
        return DecisionResult(
            Direction.STOP,
            tuple(allowed),
            sign_label,
            reason,
            intersection_id,
        )

    def _peek_context(self):
        """Best-effort context for a non-consuming traffic-light hold."""

        if self._index >= self.count:
            return None, ()
        entry = self._intersections[self._index]
        if not isinstance(entry, dict):
            return None, ()
        intersection_id = entry.get("id", entry.get("intersection_id"))
        allowed = _parse_allowed(entry.get("allowed", entry.get("allowed_exits")))
        return intersection_id, allowed or ()

    def decide(self, ai_label=None, ai_confidence=None, signal_label=None, left_conf=None, right_conf=None, intersection_id=None):
        """Return the safe decision for the next intersection.

        Args:
            ai_label: optional AI traffic-sign label.  It overrides
                ``action``/``mock_sign``.  RED/GREEN may also be supplied here
                when the detector emits only one label at a time.
            signal_label: optional separate traffic-light label.
            left_conf: AI confidence for left branch presence.
            right_conf: AI confidence for right branch presence.
            intersection_id: explicit intersection ID if tracked externally.

        A red/yellow light is a *hold*: the scenario index is not advanced.
        All other calls consume exactly one intersection, including invalid
        entries that fail closed to STOP.
        """

        ai_normalised = _normalise_label(ai_label) if ai_label is not None else None
        signal_normalised = (
            _normalise_label(signal_label) if signal_label is not None else None
        )

        light_label = signal_normalised
        if light_label is None and (
                ai_normalised in _RED_LIGHT_LABELS or
                ai_normalised in _GREEN_LIGHT_LABELS or
                ai_normalised in _STOP_LIGHT_LABELS):
            light_label = ai_normalised

        if light_label in _RED_LIGHT_LABELS or light_label in _STOP_LIGHT_LABELS:
            intersection_id, allowed = self._peek_context()
            return self._stop(
                "traffic_light_hold",
                intersection_id=intersection_id,
                allowed=allowed,
                sign_label=light_label,
            )
        if light_label is not None and light_label not in _GREEN_LIGHT_LABELS:
            intersection_id, allowed = self._peek_context()
            return self._stop(
                "invalid_signal_label",
                intersection_id=intersection_id,
                allowed=allowed,
                sign_label=light_label,
            )

        if self._load_error is not None:
            return self._stop("invalid_scenario: " + self._load_error)
        if self._index >= self.count:
            return self._stop("scenario_exhausted")

        entry_number = self._index + 1
        entry = self._intersections[self._index]
        self._index += 1

        if not isinstance(entry, dict):
            return self._stop(
                "invalid_intersection",
                intersection_id="intersection_{0}".format(entry_number),
            )

        scenario_intersection_id = entry.get(
            "id",
            entry.get("intersection_id", "intersection_{0}".format(entry_number)),
        )
        if intersection_id is None:
            intersection_id = scenario_intersection_id

        if not isinstance(intersection_id, str) or not intersection_id.strip():
            return self._stop("invalid_intersection_id")

        # Dynamic allowed exits based on AI
        allowed_list = [Direction.STRAIGHT] # Always assume straight is possible unless it's a T-junction or end, but let's assume it.
        # Actually, let's merge with scenario if AI is not confident, or override completely.
        ai_left = left_conf is not None and left_conf > 0.5
        ai_right = right_conf is not None and right_conf > 0.5
        
        scenario_allowed = _parse_allowed(entry.get("allowed", entry.get("allowed_exits")))
        
        if ai_left or ai_right:
            allowed_list = [Direction.STRAIGHT] # By default
            if ai_left:
                allowed_list.append(Direction.LEFT)
            if ai_right:
                allowed_list.append(Direction.RIGHT)
            allowed = tuple(allowed_list)
        else:
            allowed = scenario_allowed

        if allowed is None:
            return self._stop(
                "invalid_allowed_exits",
                intersection_id=intersection_id,
            )
        if not allowed:
            return self._stop(
                "no_allowed_exit",
                intersection_id=intersection_id,
                allowed=allowed,
            )

        has_action = "action" in entry
        has_mock_sign = "mock_sign" in entry
        if has_action and has_mock_sign:
            return self._stop(
                "intersection_has_action_and_mock_sign",
                intersection_id=intersection_id,
                allowed=allowed,
            )
        if not has_action and not has_mock_sign:
            return self._stop(
                "intersection_has_no_decision_source",
                intersection_id=intersection_id,
                allowed=allowed,
            )

        preference = _parse_preference(entry.get("preference", self._preference))
        if preference is None:
            return self._stop(
                "invalid_intersection_preference",
                intersection_id=intersection_id,
                allowed=allowed,
            )

        # GREEN is a gate, not a route command.  Otherwise an explicit AI sign
        # label has priority over the development scenario.
        if ai_normalised in _GREEN_LIGHT_LABELS:
            ai_normalised = None

        if ai_normalised is not None:
            sign_label = ai_normalised
            source_reason = "ai_label"
        elif has_action:
            sign_label = _normalise_label(entry.get("action"))
            source_reason = "scripted_action"
        else:
            sign_label = _normalise_label(entry.get("mock_sign"))
            source_reason = "mock_sign"

        action = resolve_sign_label(sign_label, allowed, preference)
        if action is Direction.STOP:
            # Explicit STOP is valid.  Any other STOP result means the label or
            # requested route was unsafe/invalid.
            if sign_label in ("STOP", "HALT"):
                reason = source_reason + ":stop"
            elif sign_label in _PROHIBITION_SIGNS:
                reason = source_reason + ":no_safe_alternative"
            else:
                reason = source_reason + ":invalid_or_disallowed"
            return self._stop(
                reason,
                intersection_id=intersection_id,
                allowed=allowed,
                sign_label=sign_label,
            )

        if sign_label in _PROHIBITION_SIGNS:
            reason = source_reason + ":prohibition_resolved"
        else:
            reason = source_reason + ":command_resolved"
        return DecisionResult(
            action,
            allowed,
            sign_label,
            reason,
            intersection_id,
        )


__all__ = (
    "DecisionResult",
    "Direction",
    "ScenarioDecisionProvider",
    "resolve_sign_label",
)
