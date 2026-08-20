# -*- coding: utf-8 -*-
"""Pure unit tests for the Smart City V2 finite-state controller."""

from __future__ import absolute_import

import math
import unittest

from src.smart_city.v2.config import SmartCityConfig
from src.smart_city.v2.controller import SmartCityFSM, SmartCityState
from src.smart_city.v2.decision import ScenarioDecisionProvider


class FakeClock(object):
    """Small monotonic clock controlled explicitly by each test."""

    def __init__(self, start=10.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


class FakeFrame(object):
    """Only ``shape`` is needed by the controller's geometry calculations."""

    shape = (480, 640, 3)


class FakePerception(object):
    """Minimal perception snapshot with safe lane defaults."""

    def __init__(self, **overrides):
        self.frame = FakeFrame()
        self.lane_x_near = 320.0
        self.lane_x_far = 320.0
        self.lane_confidence = 0.95
        self.stop_line = False
        self.stop_line_score = 0.0
        self.stop_line_y = None
        self.green_ahead_ratio = 0.0
        self.green_left_ratio = 0.0
        self.green_right_ratio = 0.0
        self.green_danger = False
        self.avoidance_bias = 0.0
        for name, value in overrides.items():
            setattr(self, name, value)


class SmartCityFSMTests(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.config = SmartCityConfig()

        # Short deterministic timings make boundary behavior easy to test.
        self.config.stop_confirm_frames = 3
        self.config.initial_lane_stable_frames = 1
        self.config.ai_confirm_frames = 1
        self.config.stop_hold_seconds = 0.10
        self.config.lane_loss_stop_seconds = 0.10
        self.config.lane_loss_estop_seconds = 0.30
        self.config.nudge_left_seconds = 0.10
        self.config.nudge_right_seconds = 0.10
        self.config.turn_min_seconds = 0.10
        self.config.turn_nominal_seconds = 0.20
        self.config.turn_max_seconds = 0.40
        self.config.straight_cross_seconds = 0.15
        self.config.reacquire_timeout_seconds = 0.30
        self.config.reacquire_stable_frames = 2
        self.config.intersection_cooldown_seconds = 0.20
        self.config.exit_clear_frames = 2
        self.config.exit_lockout_max_seconds = 0.80
        self.config.red_light_timeout_seconds = 1.00
        self.seq = 0

    @staticmethod
    def scenario_entry(action="RIGHT", allowed=None, mock_sign=None):
        entry = {
            "id": "test_intersection",
            "allowed": allowed or ["LEFT", "STRAIGHT", "RIGHT"],
        }
        if mock_sign is None:
            entry["action"] = action
        else:
            entry["mock_sign"] = mock_sign
        return entry

    def make_fsm(self, intersections=None):
        if intersections is None:
            intersections = [self.scenario_entry()]
        provider = ScenarioDecisionProvider({"intersections": intersections})
        fsm = SmartCityFSM(provider, config=self.config, clock=self.clock)
        return fsm, provider

    def new_frame(self, fsm, perception=None, **kwargs):
        self.seq += 1
        if perception is None:
            perception = FakePerception()
        return fsm.update(
            perception,
            now=self.clock(),
            frame_seq=self.seq,
            **kwargs
        )

    def arm_and_acquire_lane(self, fsm):
        self.assertTrue(fsm.arm(now=self.clock()))
        command = self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.LANE_FOLLOW, fsm.state)
        self.assertGreater(command.throttle, 0.0)
        return command

    def enter_stop_hold(self, fsm):
        self.arm_and_acquire_lane(fsm)
        stop = FakePerception(
            stop_line=True,
            stop_line_score=0.95,
            stop_line_y=360,
        )

        # The first candidate moves to APPROACH_LINE and counts as frame one.
        self.new_frame(fsm, stop)
        self.assertEqual(SmartCityState.APPROACH_LINE, fsm.state)
        self.new_frame(fsm, stop)
        self.assertEqual(SmartCityState.APPROACH_LINE, fsm.state)
        command = self.new_frame(fsm, stop)

        self.assertEqual(SmartCityState.STOP_HOLD, fsm.state)
        self.assertEqual(0.0, command.throttle)
        return command

    def enter_wait_decision(self, fsm):
        self.enter_stop_hold(fsm)
        self.clock.advance(self.config.stop_hold_seconds + 0.01)
        command = self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.WAIT_DECISION, fsm.state)
        self.assertEqual(0.0, command.throttle)
        return command

    def test_disarmed_is_stationary_even_with_valid_lane(self):
        fsm, unused_provider = self.make_fsm()

        command = self.new_frame(fsm, FakePerception())

        self.assertEqual(SmartCityState.DISARMED, fsm.state)
        self.assertEqual(0.0, command.steering)
        self.assertEqual(0.0, command.throttle)
        self.assertEqual("not_armed", command.reason)

    def test_arm_requires_a_new_valid_lane_frame_before_moving(self):
        fsm, unused_provider = self.make_fsm()

        self.assertTrue(fsm.arm(now=self.clock()))
        invalid = FakePerception(
            lane_x_near=None,
            lane_x_far=None,
            lane_confidence=0.0,
        )
        waiting = self.new_frame(fsm, invalid)
        self.assertEqual(SmartCityState.WAIT_SENSORS, fsm.state)
        self.assertEqual(0.0, waiting.throttle)

        ready = self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.LANE_FOLLOW, fsm.state)
        self.assertGreater(ready.throttle, 0.0)

    def test_initial_lane_requires_several_new_frames_when_configured(self):
        self.config.initial_lane_stable_frames = 3
        fsm, unused_provider = self.make_fsm()
        self.assertTrue(fsm.arm(now=self.clock()))

        first = self.new_frame(fsm, FakePerception())
        second = self.new_frame(fsm, FakePerception())
        self.assertEqual(0.0, first.throttle)
        self.assertEqual(0.0, second.throttle)
        third = self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.LANE_FOLLOW, fsm.state)
        self.assertGreater(third.throttle, 0.0)

    def test_sensor_watchdogs_fail_closed(self):
        fsm, unused_provider = self.make_fsm()
        self.arm_and_acquire_lane(fsm)

        stale = self.new_frame(
            fsm,
            FakePerception(),
            camera_age_seconds=self.config.camera_timeout_seconds + 0.01,
        )
        self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
        self.assertEqual(0.0, stale.throttle)

        fsm.reset_stop(now=self.clock())
        self.arm_and_acquire_lane(fsm)
        collision = self.new_frame(
            fsm,
            FakePerception(),
            obstacle_distance_m=self.config.lidar_stop_distance_m,
        )
        self.assertEqual(SmartCityState.E_STOP, fsm.state)
        self.assertEqual(0.0, collision.throttle)

        fsm.reset_stop(now=self.clock())
        self.arm_and_acquire_lane(fsm)
        invalid = self.new_frame(
            fsm,
            FakePerception(),
            obstacle_distance_m=float("nan"),
        )
        self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
        self.assertEqual(0.0, invalid.throttle)

    def test_frozen_frame_sequence_cannot_satisfy_stop_line_debounce(self):
        fsm, unused_provider = self.make_fsm()
        self.arm_and_acquire_lane(fsm)
        stop = FakePerception(
            stop_line=True,
            stop_line_score=0.95,
            stop_line_y=360,
        )

        self.seq += 1
        frozen_seq = self.seq
        fsm.update(stop, now=self.clock(), frame_seq=frozen_seq)
        self.assertEqual(SmartCityState.APPROACH_LINE, fsm.state)

        for unused in range(10):
            fsm.update(stop, now=self.clock(), frame_seq=frozen_seq)
        self.assertEqual(SmartCityState.APPROACH_LINE, fsm.state)
        self.assertEqual(1, fsm._stop_streak)

        self.new_frame(fsm, stop)
        self.assertEqual(SmartCityState.APPROACH_LINE, fsm.state)
        self.new_frame(fsm, stop)
        self.assertEqual(SmartCityState.STOP_HOLD, fsm.state)

    def test_stop_line_requires_three_distinct_frames(self):
        fsm, provider = self.make_fsm()

        self.enter_stop_hold(fsm)

        self.assertEqual(3, fsm._stop_streak)
        self.assertEqual(0, provider.consumed)

    def test_one_close_white_row_then_loss_does_not_consume_route(self):
        fsm, provider = self.make_fsm()
        self.arm_and_acquire_lane(fsm)
        glare = FakePerception(
            stop_line=True,
            stop_line_score=0.95,
            stop_line_y=390,
        )
        self.new_frame(fsm, glare)
        self.assertEqual(SmartCityState.APPROACH_LINE, fsm.state)

        self.new_frame(fsm, FakePerception())
        self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.LANE_FOLLOW, fsm.state)
        self.assertEqual(0, provider.consumed)

    def test_non_finite_perception_cannot_become_full_control(self):
        fsm, unused_provider = self.make_fsm()
        self.arm_and_acquire_lane(fsm)
        command = self.new_frame(
            fsm,
            FakePerception(avoidance_bias=float("nan")),
        )
        self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
        self.assertEqual(0.0, command.steering)
        self.assertEqual(0.0, command.throttle)

    def test_red_holds_without_consuming_and_green_continues(self):
        fsm, provider = self.make_fsm([
            self.scenario_entry(action="RIGHT", allowed=["LEFT", "RIGHT"])
        ])
        self.enter_wait_decision(fsm)

        red = self.new_frame(
            fsm,
            FakePerception(),
            signal_label="RED",
            signal_confidence=0.99,
        )
        self.assertEqual(SmartCityState.WAIT_DECISION, fsm.state)
        self.assertEqual(0.0, red.throttle)
        self.assertEqual("traffic_light_hold", red.reason)
        self.assertEqual(0, provider.consumed)

        second_red = self.new_frame(
            fsm,
            FakePerception(),
            signal_label="RED_LIGHT",
            signal_confidence=0.99,
        )
        self.assertEqual(0.0, second_red.throttle)
        self.assertEqual(0, provider.consumed)

        green = self.new_frame(
            fsm,
            FakePerception(),
            signal_label="GREEN",
            signal_confidence=0.99,
        )
        self.assertEqual(SmartCityState.NUDGE, fsm.state)
        self.assertGreater(green.throttle, 0.0)
        self.assertEqual("RIGHT", green.action)
        self.assertEqual(1, provider.consumed)

    def test_green_requires_distinct_semantic_confirmations(self):
        self.config.ai_confirm_frames = 3
        fsm, provider = self.make_fsm()
        self.enter_wait_decision(fsm)

        for semantic_seq in (101, 102):
            command = self.new_frame(
                fsm,
                FakePerception(),
                signal_label="GREEN",
                signal_confidence=0.99,
                semantic_seq=semantic_seq,
            )
            self.assertEqual(0.0, command.throttle)
            self.assertEqual(0, provider.consumed)

        accepted = self.new_frame(
            fsm,
            FakePerception(),
            signal_label="GREEN",
            signal_confidence=0.99,
            semantic_seq=103,
        )
        self.assertGreater(accepted.throttle, 0.0)
        self.assertEqual(1, provider.consumed)

    def test_semantic_confirmation_must_be_consecutive(self):
        self.config.ai_confirm_frames = 3
        fsm, provider = self.make_fsm()
        self.enter_wait_decision(fsm)

        self.new_frame(
            fsm, FakePerception(), signal_label="GREEN",
            signal_confidence=0.99, semantic_seq=201, ai_required=True,
        )
        missing = self.new_frame(
            fsm, FakePerception(), semantic_seq=202, ai_required=True,
        )
        self.assertEqual("waiting_for_ai_semantics", missing.reason)
        self.new_frame(
            fsm, FakePerception(), signal_label="GREEN",
            signal_confidence=0.99, semantic_seq=203, ai_required=True,
        )
        red = self.new_frame(
            fsm, FakePerception(), signal_label="RED",
            signal_confidence=0.99, semantic_seq=204, ai_required=True,
        )
        self.assertEqual("traffic_light_hold", red.reason)
        self.assertEqual(0, provider.consumed)

        for semantic_seq in (205, 206):
            held = self.new_frame(
                fsm, FakePerception(), signal_label="GREEN",
                signal_confidence=0.99, semantic_seq=semantic_seq,
                ai_required=True,
            )
            self.assertEqual(0.0, held.throttle)
            self.assertEqual(0, provider.consumed)
        accepted = self.new_frame(
            fsm, FakePerception(), signal_label="GREEN",
            signal_confidence=0.99, semantic_seq=207, ai_required=True,
        )
        self.assertGreater(accepted.throttle, 0.0)
        self.assertEqual(1, provider.consumed)

    def test_invalid_or_interrupted_optional_semantics_never_fall_back(self):
        self.config.ai_confirm_frames = 3
        fsm, provider = self.make_fsm()
        self.enter_wait_decision(fsm)

        invalid = self.new_frame(
            fsm, FakePerception(), ai_label="   ", ai_confidence=0.99,
            semantic_seq=300,
        )
        self.assertEqual("invalid_semantic_label", invalid.reason)
        self.assertEqual(0, provider.consumed)

        first = self.new_frame(
            fsm, FakePerception(), signal_label="GREEN",
            signal_confidence=0.99, semantic_seq=301,
        )
        self.assertEqual(0.0, first.throttle)
        missing = self.new_frame(
            fsm, FakePerception(), semantic_seq=302,
        )
        self.assertEqual("waiting_for_ai_semantics", missing.reason)
        self.assertEqual(0, provider.consumed)

    def test_conflicting_or_old_semantics_do_not_consume_route(self):
        self.config.ai_confirm_frames = 1
        fsm, provider = self.make_fsm()
        self.enter_wait_decision(fsm)
        threshold = fsm._wait_decision_frame_seq

        old = self.new_frame(
            fsm, FakePerception(), signal_label="GREEN",
            signal_confidence=0.99, semantic_seq=400,
            semantic_source_frame_seq=threshold,
        )
        self.assertEqual("semantic_predates_intersection", old.reason)
        self.assertEqual(0, provider.consumed)

        conflict = self.new_frame(
            fsm, FakePerception(), ai_label="RED", ai_confidence=0.99,
            signal_label="GREEN", signal_confidence=0.99,
            semantic_seq=401,
            semantic_source_frame_seq=self.seq,
        )
        self.assertEqual("conflicting_traffic_light_labels", conflict.reason)
        self.assertEqual(0, provider.consumed)

    def test_non_finite_time_latches_safe_stop(self):
        fsm, unused_provider = self.make_fsm()
        self.arm_and_acquire_lane(fsm)
        command = fsm.update(
            FakePerception(), now=float("nan"), frame_seq=self.seq + 1
        )
        self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
        self.assertEqual("invalid_monotonic_time", command.reason)
        self.assertEqual(0.0, command.throttle)

    def test_stop_mid_intersection_requires_route_relocalization(self):
        fsm, provider = self.make_fsm()
        self.enter_wait_decision(fsm)
        moving = self.new_frame(fsm, FakePerception())
        self.assertGreater(moving.throttle, 0.0)
        self.assertEqual(1, provider.consumed)

        fsm.emergency_stop("test", now=self.clock())
        self.assertFalse(fsm.reset_stop(now=self.clock()))
        self.assertFalse(fsm.arm(now=self.clock()))
        fsm.restart_route_from_start(now=self.clock())
        self.assertEqual(0, provider.consumed)
        self.assertTrue(fsm.arm(now=self.clock()))

    def test_banned_only_exit_and_invalid_action_enter_safe_stop(self):
        unsafe_entries = (
            self.scenario_entry(allowed=["LEFT"], mock_sign="NO_LEFT"),
            self.scenario_entry(action="RIGHT", allowed=["LEFT"]),
            self.scenario_entry(allowed=["LEFT"], mock_sign="UNKNOWN_SIGN"),
        )

        for entry in unsafe_entries:
            with self.subTest(entry=entry):
                fsm, provider = self.make_fsm([entry])
                self.enter_wait_decision(fsm)
                command = self.new_frame(fsm, FakePerception())

                self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
                self.assertEqual(0.0, command.steering)
                self.assertEqual(0.0, command.throttle)
                self.assertTrue(command.reason.startswith("decision_stop:"))
                self.assertEqual(1, provider.consumed)

    def test_emergency_stop_is_latched_from_every_state(self):
        states = (
            SmartCityState.DISARMED,
            SmartCityState.WAIT_SENSORS,
            SmartCityState.LANE_FOLLOW,
            SmartCityState.APPROACH_LINE,
            SmartCityState.STOP_HOLD,
            SmartCityState.WAIT_DECISION,
            SmartCityState.NUDGE,
            SmartCityState.TURNING,
            SmartCityState.CROSSING,
            SmartCityState.REACQUIRE,
            SmartCityState.EXIT_LOCKOUT,
            SmartCityState.FINISHED,
            SmartCityState.SAFE_STOP,
            SmartCityState.E_STOP,
        )

        for state in states:
            with self.subTest(state=state):
                fsm, unused_provider = self.make_fsm()
                fsm.state = state
                fsm.armed = True

                fsm.emergency_stop("test_estop", now=self.clock())
                command = fsm.update(
                    None,
                    now=self.clock(),
                    frame_seq=self.seq,
                    camera_age_seconds=99.0,
                    obstacle_distance_m=0.0,
                )

                self.assertEqual(SmartCityState.E_STOP, fsm.state)
                self.assertFalse(fsm.armed)
                self.assertEqual(0.0, command.steering)
                self.assertEqual(0.0, command.throttle)
                self.assertEqual("test_estop", command.reason)

                self.assertTrue(fsm.reset_stop(now=self.clock()))
                self.assertEqual(SmartCityState.DISARMED, fsm.state)

    def test_lane_loss_stops_immediately_then_latches_safe_stop(self):
        fsm, unused_provider = self.make_fsm()
        self.arm_and_acquire_lane(fsm)
        missing = FakePerception(
            lane_x_near=None,
            lane_x_far=None,
            lane_confidence=0.0,
        )

        first = self.new_frame(fsm, missing)
        self.assertEqual(SmartCityState.LANE_FOLLOW, fsm.state)
        self.assertEqual(0.0, first.throttle)
        self.assertEqual("lane_temporarily_missing", first.reason)

        self.clock.advance(self.config.lane_loss_stop_seconds + 0.01)
        guarded = self.new_frame(fsm, missing)
        self.assertEqual(SmartCityState.LANE_FOLLOW, fsm.state)
        self.assertEqual(0.0, guarded.throttle)
        self.assertEqual("lane_loss_guard", guarded.reason)

        elapsed = (
            self.config.lane_loss_estop_seconds
            - self.config.lane_loss_stop_seconds
            + 0.01
        )
        self.clock.advance(elapsed)
        stopped = self.new_frame(fsm, missing)
        self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
        self.assertFalse(fsm.armed)
        self.assertEqual(0.0, stopped.throttle)
        self.assertEqual("lane_lost", stopped.reason)

    def test_turn_timeout_enters_safe_stop(self):
        fsm, unused_provider = self.make_fsm([
            self.scenario_entry(action="RIGHT", allowed=["LEFT", "RIGHT"])
        ])
        self.enter_wait_decision(fsm)
        accepted = self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.NUDGE, fsm.state)
        self.assertGreater(accepted.throttle, 0.0)

        self.clock.advance(self.config.nudge_right_seconds + 0.01)
        self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.TURNING, fsm.state)

        self.clock.advance(self.config.turn_max_seconds + 0.01)
        stopped = self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
        self.assertEqual(0.0, stopped.throttle)
        self.assertEqual("turn_timeout", stopped.reason)

    def test_reacquire_timeout_enters_safe_stop(self):
        fsm, unused_provider = self.make_fsm([
            self.scenario_entry(action="STRAIGHT", allowed=["STRAIGHT"])
        ])
        self.enter_wait_decision(fsm)
        self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.CROSSING, fsm.state)

        self.clock.advance(self.config.straight_cross_seconds + 0.01)
        self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.REACQUIRE, fsm.state)

        self.clock.advance(self.config.reacquire_timeout_seconds + 0.01)
        stopped = self.new_frame(
            fsm,
            FakePerception(
                lane_x_near=None,
                lane_x_far=None,
                lane_confidence=0.0,
            ),
        )
        self.assertEqual(SmartCityState.SAFE_STOP, fsm.state)
        self.assertEqual(0.0, stopped.throttle)
        self.assertEqual("lane_reacquire_timeout", stopped.reason)

    def test_exit_lockout_rejects_duplicate_stop_event_until_line_clears(self):
        fsm, provider = self.make_fsm([
            self.scenario_entry(action="STRAIGHT", allowed=["STRAIGHT"]),
            {
                "id": "must_not_be_consumed_during_lockout",
                "allowed": ["STRAIGHT"],
                "action": "END",
            },
        ])
        self.enter_wait_decision(fsm)
        self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.CROSSING, fsm.state)
        self.assertEqual(1, provider.consumed)

        self.clock.advance(self.config.straight_cross_seconds + 0.01)
        self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.REACQUIRE, fsm.state)

        self.new_frame(fsm, FakePerception())
        self.new_frame(fsm, FakePerception())
        self.assertEqual(SmartCityState.EXIT_LOCKOUT, fsm.state)

        duplicate_line = FakePerception(
            stop_line=True,
            stop_line_score=0.95,
            stop_line_y=360,
        )
        self.clock.advance(self.config.intersection_cooldown_seconds + 0.01)
        for unused in range(4):
            command = self.new_frame(fsm, duplicate_line)
            self.assertEqual(SmartCityState.EXIT_LOCKOUT, fsm.state)
            self.assertEqual("intersection_exit_lockout", command.reason)
            self.assertEqual(1, provider.consumed)

        for unused in range(self.config.exit_clear_frames):
            command = self.new_frame(fsm, FakePerception())

        self.assertEqual(SmartCityState.LANE_FOLLOW, fsm.state)
        self.assertEqual("intersection_cleared", command.reason)
        self.assertEqual(1, provider.consumed)


if __name__ == "__main__":
    unittest.main()
