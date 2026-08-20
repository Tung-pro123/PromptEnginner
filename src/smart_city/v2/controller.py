# -*- coding: utf-8 -*-
"""Dependency-free finite-state controller for the Smart City course.

AI is deliberately outside the steering loop.  It may report only semantic
traffic-light/sign labels.  This module validates that label against the
scripted physical exits and owns every throttle/steering decision.
"""

from __future__ import division

import math
import time

from .config import SmartCityConfig
from .decision import Direction


class SmartCityState(object):
    DISARMED = "DISARMED"
    WAIT_SENSORS = "WAIT_SENSORS"
    LANE_FOLLOW = "LANE_FOLLOW"
    APPROACH_LINE = "APPROACH_LINE"
    STOP_HOLD = "STOP_HOLD"
    WAIT_DECISION = "WAIT_DECISION"
    NUDGE = "NUDGE"
    TURNING = "TURNING"
    CROSSING = "CROSSING"
    REACQUIRE = "REACQUIRE"
    EXIT_LOCKOUT = "EXIT_LOCKOUT"
    FINISHED = "FINISHED"
    SAFE_STOP = "SAFE_STOP"
    E_STOP = "E_STOP_LATCHED"


class DriveCommand(object):
    """One normalised Ackermann command and its audit metadata."""

    __slots__ = (
        "steering",
        "throttle",
        "state",
        "reason",
        "action",
        "intersection_id",
        "timestamp",
    )

    def __init__(
        self,
        steering,
        throttle,
        state,
        reason="",
        action=None,
        intersection_id=None,
        timestamp=None,
    ):
        self.steering = float(steering)
        self.throttle = float(throttle)
        self.state = state
        self.reason = reason
        self.action = action
        self.intersection_id = intersection_id
        self.timestamp = timestamp

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


class SmartCityFSM(object):
    """Fail-closed Smart City state machine.

    ``update`` must receive monotonic time.  ``frame_seq`` should increase only
    when a genuinely new camera frame arrives; this prevents a frozen image
    from satisfying stop-line or lane-stability debounces.
    """

    def __init__(self, decision_provider, config=None, clock=None):
        self.config = config or SmartCityConfig()
        if hasattr(self.config, "validate"):
            self.config.validate()
        self.decision_provider = decision_provider
        self.clock = clock or time.monotonic

        now = self.clock()
        self.state = SmartCityState.DISARMED
        self.state_since = now
        self.armed = False
        self.stop_reason = "not_armed"

        self._last_frame_seq = None
        self._last_control_time = None
        self._previous_error = None
        self._last_steering = 0.0
        self._lane_missing_since = None
        self._initial_lane_streak = 0
        self._update_has_new_frame = False

        self._stop_streak = 0
        self._stop_lost_streak = 0
        self._stop_confirmed = False
        self._closest_stop_y = 0.0
        self._last_stop_y = None
        self._green_streak = 0

        self._decision = None
        self._pending_action = None
        self._turn_lane_streak = 0
        self._reacquire_streak = 0
        self._exit_clear_streak = 0
        self._intersection_latched = False
        self._wait_decision_since = None
        self._last_semantic_seq = None
        self._semantic_key = None
        self._semantic_streak = 0
        self._semantic_mode_latched = False
        self._wait_decision_frame_seq = None
        self._resume_requires_relocalization = False

    def arm(self, now=None):
        """Arm from DISARMED only.  A latched stop must be reset explicitly."""
        now = self.clock() if now is None else float(now)
        if (
            self.state != SmartCityState.DISARMED
            or self._resume_requires_relocalization
        ):
            return False
        self.armed = True
        self.stop_reason = "waiting_for_valid_lane"
        self._initial_lane_streak = 0
        self._transition(SmartCityState.WAIT_SENSORS, now)
        return True

    def disarm(self, reason="operator_disarm", now=None):
        now = self.clock() if now is None else float(now)
        if self._intersection_latched or self._decision is not None:
            self._resume_requires_relocalization = True
        self.armed = False
        self.stop_reason = reason
        if not self._resume_requires_relocalization:
            self._reset_intersection_tracking()
        self._transition(SmartCityState.DISARMED, now)

    def emergency_stop(self, reason="operator_estop", now=None):
        """Latch an emergency stop.  Subsequent updates always return zero."""
        now = self.clock() if now is None else float(now)
        if self._intersection_latched or self._decision is not None:
            self._resume_requires_relocalization = True
        self.armed = False
        self.stop_reason = reason
        self._transition(SmartCityState.E_STOP, now)

    def reset_stop(self, now=None):
        """Clear SAFE_STOP/E_STOP into DISARMED; it never re-arms the car."""
        if self.state not in (SmartCityState.SAFE_STOP, SmartCityState.E_STOP):
            return False
        if self._resume_requires_relocalization:
            return False
        now = self.clock() if now is None else float(now)
        self.armed = False
        self.stop_reason = "reset_to_disarmed"
        self._reset_intersection_tracking()
        self._transition(SmartCityState.DISARMED, now)
        return True

    def restart_route_from_start(self, now=None):
        """Explicit recovery after physically returning the car to the start.

        This is intentionally stronger than ``reset_stop``: it rewinds the
        scenario transaction.  It never arms the vehicle.
        """
        if self.armed:
            return False
        now = self.clock() if now is None else float(now)
        if hasattr(self.decision_provider, "reset"):
            self.decision_provider.reset()
        self.armed = False
        self._resume_requires_relocalization = False
        self._reset_intersection_tracking()
        self.stop_reason = "route_restarted_at_start"
        self._transition(SmartCityState.DISARMED, now)
        return True

    def update(
        self,
        perception,
        now=None,
        frame_seq=None,
        camera_age_seconds=0.0,
        obstacle_distance_m=None,
        ai_label=None,
        ai_confidence=None,
        signal_label=None,
        signal_confidence=None,
        crosswalk_conf=None,
        left_conf=None,
        right_conf=None,
        ai_required=False,
        semantic_seq=None,
        semantic_source_frame_seq=None,
    ):
        """Advance the FSM and return a :class:`DriveCommand`.

        ``ai_label`` is a sign class such as ``NO_LEFT`` or ``TURN_RIGHT``.
        ``signal_label`` is ``RED``, ``YELLOW`` or ``GREEN``.  Confidence below
        ``ai_min_confidence`` produces a stationary wait, never a guessed turn.
        Passing no AI labels activates the configured scenario/mock provider.
        """
        try:
            now = float(self.clock() if now is None else now)
        except (OverflowError, TypeError, ValueError):
            now = float("nan")
        if not math.isfinite(now):
            try:
                fallback_now = float(self.clock())
            except (OverflowError, TypeError, ValueError):
                fallback_now = 0.0
            if not math.isfinite(fallback_now):
                fallback_now = 0.0
            if self.state == SmartCityState.E_STOP:
                return self._zero(fallback_now, self.stop_reason)
            return self._safe_stop("invalid_monotonic_time", fallback_now)
        new_frame = self._is_new_frame(frame_seq)
        self._update_has_new_frame = new_frame
        if semantic_seq is None:
            semantic_seq = frame_seq

        if self.state == SmartCityState.E_STOP:
            return self._zero(now, self.stop_reason)
        if self.state == SmartCityState.FINISHED:
            return self._zero(now, "scenario_finished")
        if self.state == SmartCityState.SAFE_STOP:
            return self._zero(now, self.stop_reason)
        if not self.armed or self.state == SmartCityState.DISARMED:
            return self._zero(now, self.stop_reason or "not_armed")

        if perception is None:
            return self._safe_stop("missing_camera_frame", now)
        if camera_age_seconds is not None:
            camera_age = float(camera_age_seconds)
            if (
                not math.isfinite(camera_age)
                or camera_age < 0.0
                or camera_age > self.config.camera_timeout_seconds
            ):
                return self._safe_stop("camera_stale", now)
        if obstacle_distance_m is not None:
            obstacle_distance = float(obstacle_distance_m)
            if not math.isfinite(obstacle_distance) or obstacle_distance < 0.0:
                return self._safe_stop("invalid_lidar_distance", now)
            if obstacle_distance <= self.config.lidar_stop_distance_m:
                self.emergency_stop("collision_guard", now)
                return self._zero(now, self.stop_reason)

        lane_valid = self._lane_valid(perception)
        if self.state == SmartCityState.WAIT_SENSORS:
            if new_frame:
                if lane_valid:
                    self._initial_lane_streak += 1
                else:
                    self._initial_lane_streak = 0
            if (
                lane_valid
                and self._initial_lane_streak
                >= self.config.initial_lane_stable_frames
            ):
                self._lane_missing_since = None
                self._transition(SmartCityState.LANE_FOLLOW, now)
                return self._lane_command(perception, now, "sensors_ready")
            if now - self.state_since > self.config.sensor_acquire_timeout_seconds:
                return self._safe_stop("lane_not_acquired", now)
            return self._zero(now, "waiting_for_valid_lane")

        if self.state in (
            SmartCityState.LANE_FOLLOW,
            SmartCityState.APPROACH_LINE,
            SmartCityState.EXIT_LOCKOUT,
        ):
            lane_failure = self._check_lane_loss(lane_valid, now)
            if lane_failure is not None:
                return lane_failure
        else:
            self._lane_missing_since = None

        if self.state == SmartCityState.LANE_FOLLOW:
            return self._update_lane_follow(perception, now, new_frame, crosswalk_conf)
        if self.state == SmartCityState.APPROACH_LINE:
            return self._update_approach(perception, now, new_frame, crosswalk_conf)
        if self.state == SmartCityState.STOP_HOLD:
            if now - self.state_since >= self.config.stop_hold_seconds:
                self._wait_decision_since = now
                self._wait_decision_frame_seq = self._last_frame_seq
                self._transition(SmartCityState.WAIT_DECISION, now)
            return self._zero(now, "full_stop_before_decision")
        if self.state == SmartCityState.WAIT_DECISION:
            return self._update_decision(
                now,
                ai_label,
                ai_confidence,
                signal_label,
                signal_confidence,
                left_conf,
                right_conf,
                ai_required,
                semantic_seq,
                semantic_source_frame_seq,
                perception,
            )
        if self.state == SmartCityState.NUDGE:
            return self._update_nudge(perception, now)
        if self.state == SmartCityState.TURNING:
            return self._update_turn(perception, now, new_frame)
        if self.state == SmartCityState.CROSSING:
            return self._update_crossing(perception, now)
        if self.state == SmartCityState.REACQUIRE:
            return self._update_reacquire(perception, now, new_frame)
        if self.state == SmartCityState.EXIT_LOCKOUT:
            return self._update_exit_lockout(perception, now, new_frame, crosswalk_conf)

        return self._safe_stop("unknown_fsm_state", now)

    def _update_lane_follow(self, perception, now, new_frame, crosswalk_conf):
        if new_frame:
            if self._is_stop_candidate(perception, crosswalk_conf):
                self._record_stop(perception, detected=True)
                if self._stop_y_ratio(perception) >= 0.52:
                    self._transition(SmartCityState.APPROACH_LINE, now)
                    return self._lane_command(
                        perception,
                        now,
                        "stop_line_candidate",
                        throttle=self.config.approach_throttle,
                    )
            else:
                self._clear_stop_candidate()

            if bool(getattr(perception, "green_danger", False)):
                self._green_streak += 1
            else:
                self._green_streak = 0

        if self._green_streak >= self.config.green_danger_confirm_frames:
            return self._safe_stop("green_keepout_ahead_without_junction", now)
        return self._lane_command(perception, now, "lane_follow")

    def _update_approach(self, perception, now, new_frame, crosswalk_conf):
        if now - self.state_since > 6.5:
            return self._safe_stop("stop_line_approach_timeout", now)

        if new_frame:
            detected = self._is_stop_candidate(perception, crosswalk_conf)
            self._record_stop(perception, detected=detected)
            if self._stop_streak >= self.config.stop_confirm_frames:
                self._stop_confirmed = True

            y_ratio = self._stop_y_ratio(perception)
            close_enough = y_ratio >= 0.70 or self._closest_stop_y >= 0.76
            if self._stop_confirmed and close_enough:
                self._intersection_latched = True
                self._transition(SmartCityState.STOP_HOLD, now)
                return self._zero(now, "confirmed_stop_line")

            if not detected:
                if self._stop_confirmed:
                    if self._stop_lost_streak >= 2:
                        self._intersection_latched = True
                        self._transition(SmartCityState.STOP_HOLD, now)
                        return self._zero(now, "confirmed_line_passed_under_camera")
                elif self._stop_lost_streak >= 2:
                    self._clear_stop_candidate()
                    self._transition(SmartCityState.LANE_FOLLOW, now)

        return self._lane_command(
            perception,
            now,
            "approaching_stop_line",
            throttle=self.config.approach_throttle,
        )

    def _update_decision(
        self,
        now,
        ai_label,
        ai_confidence,
        signal_label,
        signal_confidence,
        left_conf,
        right_conf,
        ai_required,
        semantic_seq,
        semantic_source_frame_seq,
        perception,
    ):
        if now - self.state_since > self.config.red_light_timeout_seconds:
            return self._safe_stop("decision_timeout", now)

        raw_has_semantics = ai_label is not None or signal_label is not None
        if raw_has_semantics:
            self._semantic_mode_latched = True
        ai_label, ai_valid = self._normalise_semantic_label(ai_label)
        signal_label, signal_valid = self._normalise_semantic_label(signal_label)
        if not ai_valid or not signal_valid:
            self._reset_semantic_confirmation()
            return self._zero(now, "invalid_semantic_label")
        if not self._confidence_ok(ai_label, ai_confidence):
            self._reset_semantic_confirmation()
            return self._zero(now, "low_confidence_sign")
        if not self._confidence_ok(signal_label, signal_confidence):
            self._reset_semantic_confirmation()
            return self._zero(now, "low_confidence_signal")
        if (
            (ai_required or self._semantic_mode_latched)
            and ai_label is None
            and signal_label is None
        ):
            self._reset_semantic_confirmation()
            return self._zero(now, "waiting_for_ai_semantics")

        has_semantics = ai_label is not None or signal_label is not None
        ai_light = self._light_state(ai_label)
        signal_light = self._light_state(signal_label)
        if ai_light is not None and signal_light is not None and ai_light != signal_light:
            self._reset_semantic_confirmation()
            return self._zero(now, "conflicting_traffic_light_labels")
        light_state = signal_light if signal_light is not None else ai_light
        is_hold_signal = light_state in ("RED", "YELLOW")
        if is_hold_signal:
            self._reset_semantic_confirmation()
        if has_semantics and not is_hold_signal:
            if (
                semantic_source_frame_seq is not None
                and self._wait_decision_frame_seq is not None
                and semantic_source_frame_seq <= self._wait_decision_frame_seq
            ):
                self._reset_semantic_confirmation()
                return self._zero(now, "semantic_predates_intersection")
            if not self._confirm_semantics(
                ai_label, signal_label, semantic_seq
            ):
                return self._zero(now, "confirming_ai_semantics")

        try:
            result = self.decision_provider.decide(
                ai_label=ai_label,
                ai_confidence=ai_confidence,
                left_conf=left_conf,
                right_conf=right_conf,
                intersection_id=self._intersection_id,
            )
        except Exception as exc:
            return self._safe_stop(
                "decision_provider_error:%s" % exc.__class__.__name__, now
            )

        action = result.action
        action_name = getattr(action, "value", str(action)).upper()
        self._decision = result

        if action_name == "STOP":
            # A red/yellow light is a non-consuming hold.  All other STOPs are
            # invalid/final commands and remain fail-closed.
            if result.reason == "traffic_light_hold":
                return self._zero(now, "traffic_light_hold")
            return self._safe_stop("decision_stop:" + result.reason, now)
        if action_name in ("END", "FINISH", "GOAL"):
            self.armed = False
            self._transition(SmartCityState.FINISHED, now)
            return self._zero(now, "scenario_finished")
        if action_name not in ("LEFT", "RIGHT", "STRAIGHT"):
            return self._safe_stop("unsupported_decision", now)
        if (
            action_name == "STRAIGHT"
            and self._green_ahead_ratio(perception)
            >= self.config.green_danger_ratio
        ):
            return self._safe_stop("straight_exit_blocked_by_keepout", now)

        self._pending_action = action
        self._turn_lane_streak = 0
        self._reacquire_streak = 0
        if action_name == "STRAIGHT":
            self._transition(SmartCityState.CROSSING, now)
            return self._command(0.0, self.config.straight_throttle, now,
                                 "crossing_straight")

        self._transition(SmartCityState.NUDGE, now)
        return self._nudge_command(now, "decision_accepted")

    def _update_nudge(self, perception, now):
        elapsed = now - self.state_since
        action_name = self._action_name()
        duration = (
            self.config.nudge_left_seconds
            if action_name == "LEFT"
            else self.config.nudge_right_seconds
        )
        # At a T junction the green island enters the projected corridor; once
        # the axle has started moving, begin the turn instead of driving at it.
        green_close = self._green_ahead_ratio(perception)
        if green_close >= 0.48:
            return self._safe_stop("green_keepout_too_close_for_nudge", now)
        early_turn = green_close >= self.config.green_danger_ratio
        if elapsed >= duration or early_turn:
            self._turn_lane_streak = 0
            self._transition(SmartCityState.TURNING, now)
            return self._turn_command(now, 0.0, "turn_started")
        return self._nudge_command(now, "junction_nudge")

    def _update_turn(self, perception, now, new_frame):
        elapsed = now - self.state_since
        if elapsed > self.config.turn_max_seconds:
            return self._safe_stop("turn_timeout", now)

        if self._green_ahead_ratio(perception) >= 0.48:
            return self._safe_stop("green_keepout_during_turn", now)

        if new_frame and elapsed >= self.config.turn_min_seconds:
            if self._lane_centered(perception):
                self._turn_lane_streak += 1
            else:
                self._turn_lane_streak = 0

        ready_time = max(
            self.config.turn_min_seconds,
            self.config.turn_nominal_seconds * 0.90,
        )
        if (
            elapsed >= ready_time
            and self._turn_lane_streak >= self.config.turn_lane_confirm_frames
        ):
            self._reacquire_streak = 0
            self._transition(SmartCityState.REACQUIRE, now)
            return self._lane_command(
                perception,
                now,
                "new_lane_candidate",
                throttle=self.config.reacquire_throttle,
            )
        return self._turn_command(now, elapsed, "turning")

    def _update_crossing(self, perception, now):
        elapsed = now - self.state_since
        if (
            self._green_ahead_ratio(perception)
            >= self.config.green_danger_ratio
        ):
            return self._safe_stop("green_keepout_during_straight_crossing", now)
        if elapsed > self.config.straight_cross_seconds:
            self._reacquire_streak = 0
            self._transition(SmartCityState.REACQUIRE, now)
            return self._zero(now, "straight_cross_complete_check_lane")
        return self._command(
            0.0,
            self.config.straight_throttle,
            now,
            "crossing_straight",
        )

    def _update_reacquire(self, perception, now, new_frame):
        if now - self.state_since > self.config.reacquire_timeout_seconds:
            return self._safe_stop("lane_reacquire_timeout", now)
        if self._green_ahead_ratio(perception) >= self.config.green_danger_ratio:
            return self._safe_stop("green_keepout_during_reacquire", now)

        if new_frame:
            if self._lane_centered(perception):
                self._reacquire_streak += 1
            else:
                self._reacquire_streak = 0

        if self._reacquire_streak >= self.config.reacquire_stable_frames:
            self._exit_clear_streak = 0
            self._transition(SmartCityState.EXIT_LOCKOUT, now)
            return self._lane_command(
                perception,
                now,
                "lane_reacquired",
                throttle=self.config.reacquire_throttle,
            )

        if not self._lane_valid(perception):
            return self._zero(now, "waiting_for_lane_reacquire")
        return self._lane_command(
            perception,
            now,
            "reacquiring_lane",
            throttle=self.config.reacquire_throttle,
        )

    def _update_exit_lockout(self, perception, now, new_frame, crosswalk_conf):
        elapsed = now - self.state_since
        if elapsed > self.config.exit_lockout_max_seconds:
            return self._safe_stop("intersection_exit_not_cleared", now)
        if self._green_ahead_ratio(perception) >= self.config.green_danger_ratio:
            return self._safe_stop("green_keepout_during_exit", now)

        if new_frame:
            if self._is_stop_candidate(perception, crosswalk_conf):
                self._exit_clear_streak = 0
            else:
                self._exit_clear_streak += 1

        if (
            elapsed >= self.config.intersection_cooldown_seconds
            and self._exit_clear_streak >= self.config.exit_clear_frames
        ):
            self._reset_intersection_tracking()
            self._transition(SmartCityState.LANE_FOLLOW, now)
            return self._lane_command(perception, now, "intersection_cleared")
        return self._lane_command(
            perception,
            now,
            "intersection_exit_lockout",
            throttle=self.config.reacquire_throttle,
        )

    def _check_lane_loss(self, lane_valid, now):
        if lane_valid:
            self._lane_missing_since = None
            return None
        if self._lane_missing_since is None:
            self._lane_missing_since = now
            return self._zero(now, "lane_temporarily_missing")
        missing_for = now - self._lane_missing_since
        if missing_for >= self.config.lane_loss_estop_seconds:
            return self._safe_stop("lane_lost", now)
        if missing_for >= self.config.lane_loss_stop_seconds:
            return self._zero(now, "lane_loss_guard")
        return self._zero(now, "lane_temporarily_missing")

    def _lane_command(self, perception, now, reason, throttle=None):
        if throttle is None:
            throttle = self.config.cruise_throttle
        if not self._update_has_new_frame and self._last_control_time is not None:
            speed_scale = 1.0 - 0.45 * min(1.0, abs(self._last_steering))
            return self._command(
                self._last_steering, throttle * speed_scale, now, reason
            )

        width = float(perception.frame.shape[1])
        centre = width * 0.5
        near_x = float(perception.lane_x_near)
        far_x = perception.lane_x_far
        error = (near_x - centre) / max(1.0, centre)
        heading = 0.0
        if far_x is not None:
            heading = (near_x - float(far_x)) / max(1.0, centre)

        dt = 1.0 / max(1.0, self.config.loop_hz)
        if self._last_control_time is not None:
            dt = max(0.01, min(0.20, now - self._last_control_time))
        derivative = 0.0
        if self._previous_error is not None:
            derivative = (error - self._previous_error) / dt
        steering = (
            self.config.lane_kp * error
            + self.config.lane_kd * derivative
            + self.config.heading_gain * heading
            + self.config.green_avoidance_gain
            * float(getattr(perception, "avoidance_bias", 0.0))
        )
        steering = self._clamp(
            steering,
            -self.config.max_lane_steering,
            self.config.max_lane_steering,
        )
        max_step = self.config.steering_slew_per_second * dt
        steering = self._clamp(
            steering,
            self._last_steering - max_step,
            self._last_steering + max_step,
        )

        self._previous_error = error
        self._last_control_time = now
        self._last_steering = steering
        # Slow further for large corrections; never increase over the caller's
        # phase-specific throttle.
        speed_scale = 1.0 - 0.45 * min(1.0, abs(steering))
        return self._command(steering, throttle * speed_scale, now, reason)

    def _nudge_command(self, now, reason):
        full = self._turn_steering()
        return self._command(full * 0.34, self.config.turn_throttle, now, reason)

    def _turn_command(self, now, elapsed, reason):
        ramp = self._clamp(elapsed / 0.28, 0.55, 1.0)
        throttle = self.config.turn_throttle
        if elapsed > self.config.turn_nominal_seconds + 0.25:
            throttle = min(throttle, self.config.reacquire_throttle)
        return self._command(
            self._turn_steering() * ramp,
            throttle,
            now,
            reason,
        )

    def _turn_steering(self):
        if self._action_name() == "LEFT":
            return self.config.turn_steering_left
        return self.config.turn_steering_right

    def _action_name(self):
        return getattr(self._pending_action, "value", str(self._pending_action)).upper()

    def _lane_valid(self, perception):
        near = getattr(perception, "lane_x_near", None)
        if near is None:
            return False
        try:
            near = float(near)
            confidence = float(getattr(perception, "lane_confidence", 0.0))
        except (OverflowError, TypeError, ValueError):
            return False
        return (
            math.isfinite(near)
            and math.isfinite(confidence)
            and confidence >= self.config.lane_min_confidence
            and confidence <= 1.0
        )

    @staticmethod
    def _green_ahead_ratio(perception):
        try:
            value = float(getattr(perception, "green_ahead_ratio", 0.0))
        except (OverflowError, TypeError, ValueError):
            return 1.0
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return 1.0
        return value

    def _lane_centered(self, perception):
        if not self._lane_valid(perception):
            return False
        width = float(perception.frame.shape[1])
        if not math.isfinite(width) or width <= 0.0:
            return False
        centre = width * 0.5
        near_error = abs(float(perception.lane_x_near) - centre) / centre
        far_x = getattr(perception, "lane_x_far", None)
        if far_x is None:
            return False
        try:
            far_value = float(far_x)
        except (OverflowError, TypeError, ValueError):
            return False
        if not math.isfinite(far_value):
            return False
        heading_error = abs(float(perception.lane_x_near) - far_value) / centre
        return (
            near_error <= self.config.reacquire_center_error_ratio
            and heading_error <= self.config.reacquire_heading_error_ratio
        )

    def _is_stop_candidate(self, perception, crosswalk_conf=None):
        try:
            score = float(getattr(perception, "stop_line_score", 0.0))
        except (OverflowError, TypeError, ValueError):
            return False
            
        opencv_stop = (
            bool(getattr(perception, "stop_line", False))
            and math.isfinite(score)
            and 0.55 <= score <= 1.0
        )
        
        ai_stop = crosswalk_conf is not None and crosswalk_conf >= 0.5
        
        return opencv_stop or ai_stop

    def _record_stop(self, perception, detected):
        if detected:
            y_ratio = self._stop_y_ratio(perception)
            tolerance = self.config.stop_y_backtrack_tolerance_ratio
            if (
                self._last_stop_y is not None
                and y_ratio + tolerance < self._last_stop_y
            ):
                # A row jumping away from the camera is likely glare/a frame
                # from another object, so restart temporal confirmation.
                self._stop_streak = 1
            else:
                self._stop_streak += 1
            self._stop_lost_streak = 0
            self._last_stop_y = y_ratio
            self._closest_stop_y = max(
                self._closest_stop_y, y_ratio
            )
        else:
            self._stop_lost_streak += 1
            if not self._stop_confirmed:
                self._stop_streak = 0

    def _clear_stop_candidate(self):
        self._stop_streak = 0
        self._stop_lost_streak = 0
        self._stop_confirmed = False
        self._closest_stop_y = 0.0
        self._last_stop_y = None

    @staticmethod
    def _stop_y_ratio(perception):
        y = getattr(perception, "stop_line_y", None)
        if y is None:
            return 0.0
        height = float(perception.frame.shape[0])
        return float(y) / max(1.0, height)

    def _reset_intersection_tracking(self):
        self._clear_stop_candidate()
        self._green_streak = 0
        self._decision = None
        self._pending_action = None
        self._turn_lane_streak = 0
        self._reacquire_streak = 0
        self._exit_clear_streak = 0
        self._intersection_latched = False
        self._wait_decision_since = None
        self._last_semantic_seq = None
        self._semantic_key = None
        self._semantic_streak = 0
        self._semantic_mode_latched = False
        self._wait_decision_frame_seq = None

    def _is_new_frame(self, frame_seq):
        if frame_seq is None:
            return True
        if frame_seq == self._last_frame_seq:
            return False
        self._last_frame_seq = frame_seq
        return True

    def _confidence_ok(self, label, confidence):
        if label is None:
            return True
        if confidence is None:
            return False
        try:
            value = float(confidence)
        except (OverflowError, TypeError, ValueError):
            return False
        return (
            math.isfinite(value)
            and value >= self.config.ai_min_confidence
            and value <= 1.0
        )

    @staticmethod
    def _normalise_semantic_label(label):
        if label is None:
            return None, True
        if not isinstance(label, str):
            return None, False
        value = label.strip().upper().replace("-", "_").replace(" ", "_")
        while "__" in value:
            value = value.replace("__", "_")
        if not value:
            return None, False
        return value, True

    @staticmethod
    def _light_state(label):
        if label in ("RED", "RED_LIGHT", "TRAFFIC_LIGHT_RED"):
            return "RED"
        if label in ("YELLOW", "YELLOW_LIGHT", "TRAFFIC_LIGHT_YELLOW"):
            return "YELLOW"
        if label in ("GREEN", "GREEN_LIGHT", "TRAFFIC_LIGHT_GREEN"):
            return "GREEN"
        return None

    def _confirm_semantics(self, ai_label, signal_label, semantic_seq):
        key = (
            None if ai_label is None else str(ai_label).strip().upper(),
            None if signal_label is None else str(signal_label).strip().upper(),
        )
        if semantic_seq == self._last_semantic_seq:
            return self._semantic_streak >= self.config.ai_confirm_frames
        self._last_semantic_seq = semantic_seq
        if key == self._semantic_key:
            self._semantic_streak += 1
        else:
            self._semantic_key = key
            self._semantic_streak = 1
        return self._semantic_streak >= self.config.ai_confirm_frames

    def _reset_semantic_confirmation(self):
        self._last_semantic_seq = None
        self._semantic_key = None
        self._semantic_streak = 0

    def _safe_stop(self, reason, now):
        if self._intersection_latched or self._decision is not None:
            self._resume_requires_relocalization = True
        self.armed = False
        self.stop_reason = reason
        self._transition(SmartCityState.SAFE_STOP, now)
        return self._zero(now, reason)

    def _transition(self, state, now):
        if state != self.state and state in (
            SmartCityState.STOP_HOLD,
            SmartCityState.WAIT_DECISION,
            SmartCityState.NUDGE,
            SmartCityState.TURNING,
            SmartCityState.CROSSING,
            SmartCityState.REACQUIRE,
        ):
            self._previous_error = None
            self._last_control_time = None
        self.state = state
        self.state_since = now

    def _zero(self, now, reason):
        self._last_steering = 0.0
        return self._command(0.0, 0.0, now, reason)

    def _command(self, steering, throttle, now, reason):
        steering = float(steering)
        throttle = float(throttle)
        if not math.isfinite(steering) or not math.isfinite(throttle):
            steering = 0.0
            throttle = 0.0
            self.armed = False
            self.stop_reason = "non_finite_control"
            self.state = SmartCityState.SAFE_STOP
            self.state_since = now
            if self._intersection_latched or self._decision is not None:
                self._resume_requires_relocalization = True
            reason = self.stop_reason
        steering = self._clamp(steering, -1.0, 1.0)
        throttle = self._clamp(throttle, 0.0, 1.0)
        self._last_steering = steering
        action = self._action_name() if self._pending_action is not None else None
        intersection_id = None
        if self._decision is not None:
            intersection_id = self._decision.intersection_id
        return DriveCommand(
            steering,
            throttle,
            self.state,
            reason=reason,
            action=action,
            intersection_id=intersection_id,
            timestamp=now,
        )

    @staticmethod
    def _clamp(value, lower, upper):
        if not math.isfinite(float(value)):
            return float("nan")
        return max(lower, min(upper, value))


__all__ = ("DriveCommand", "SmartCityFSM", "SmartCityState")
