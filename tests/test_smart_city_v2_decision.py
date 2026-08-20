# -*- coding: utf-8 -*-
"""Unit tests for the dependency-free Smart City V2 decision layer."""

from __future__ import absolute_import

import os
import unittest

from src.smart_city.v2.decision import (
    Direction,
    ScenarioDecisionProvider,
    resolve_sign_label,
)


def make_scenario(intersections, preference=None):
    scenario = {"intersections": intersections}
    if preference is not None:
        scenario["preference"] = preference
    return scenario


class ResolveSignLabelTests(unittest.TestCase):

    def test_turn_commands_must_be_allowed(self):
        allowed = ["LEFT", "STRAIGHT"]
        self.assertEqual(
            Direction.LEFT,
            resolve_sign_label("TURN_LEFT", allowed),
        )
        self.assertEqual(
            Direction.STOP,
            resolve_sign_label("TURN_RIGHT", allowed),
        )

    def test_prohibition_uses_preference_and_never_leaves_allowed_set(self):
        allowed = ["LEFT", "RIGHT"]
        action = resolve_sign_label(
            "NO_LEFT",
            allowed,
            preference=["STRAIGHT", "LEFT", "RIGHT"],
        )
        self.assertEqual(Direction.RIGHT, action)
        self.assertIn(action.value, allowed)

    def test_prohibition_with_no_alternative_stops(self):
        self.assertEqual(
            Direction.STOP,
            resolve_sign_label("NO_RIGHT", ["RIGHT"]),
        )

    def test_unknown_or_malformed_input_stops(self):
        self.assertEqual(
            Direction.STOP,
            resolve_sign_label("UNKNOWN_CLASS", ["LEFT"]),
        )
        self.assertEqual(
            Direction.STOP,
            resolve_sign_label("TURN_LEFT", ["GREEN_ZONE"]),
        )

    def test_terminal_aliases_resolve_to_end(self):
        for label in ("END", "FINISH", "GOAL"):
            with self.subTest(label=label):
                self.assertEqual(
                    Direction.END,
                    resolve_sign_label(label, ["STRAIGHT"]),
                )


class ScenarioDecisionProviderTests(unittest.TestCase):

    def test_mock_sign_and_scripted_action_are_sequential(self):
        provider = ScenarioDecisionProvider(make_scenario([
            {
                "id": "crossroad",
                "allowed": ["LEFT", "STRAIGHT", "RIGHT"],
                "mock_sign": "TURN_STRAIGHT",
            },
            {
                "id": "t_junction",
                "allowed": ["LEFT", "RIGHT"],
                "action": "RIGHT",
            },
        ]))

        first = provider.decide()
        second = provider.decide()

        self.assertEqual(Direction.STRAIGHT, first.action)
        self.assertEqual("crossroad", first.intersection_id)
        self.assertEqual("TURN_STRAIGHT", first.sign_label)
        self.assertEqual(Direction.RIGHT, second.action)
        self.assertEqual("t_junction", second.intersection_id)
        self.assertEqual(0, provider.remaining)
        self.assertEqual(2, provider.consumed)
        self.assertEqual(2, provider.count)

    def test_ai_label_overrides_mock_and_scripted_sources(self):
        provider = ScenarioDecisionProvider(make_scenario([
            {
                "id": "mock",
                "allowed": ["LEFT", "RIGHT"],
                "mock_sign": "TURN_LEFT",
            },
            {
                "id": "script",
                "allowed": ["LEFT", "STRAIGHT"],
                "action": "LEFT",
            },
        ]))

        self.assertEqual(Direction.RIGHT, provider.decide("TURN_RIGHT").action)
        self.assertEqual(Direction.STRAIGHT, provider.decide("TURN_STRAIGHT").action)

    def test_ai_prohibition_respects_physical_exits(self):
        provider = ScenarioDecisionProvider(make_scenario([
            {
                "id": "three_way",
                "allowed": ["LEFT", "RIGHT"],
                "mock_sign": "TURN_LEFT",
                "preference": ["STRAIGHT", "RIGHT", "LEFT"],
            }
        ]))

        result = provider.decide(ai_label="NO_LEFT")

        self.assertEqual(Direction.RIGHT, result.action)
        self.assertEqual((Direction.LEFT, Direction.RIGHT), result.allowed)

    def test_red_light_holds_without_consuming_then_green_continues(self):
        provider = ScenarioDecisionProvider(make_scenario([
            {
                "id": "signalised_crossroad",
                "allowed": ["STRAIGHT", "RIGHT"],
                "action": "RIGHT",
            }
        ]))

        first_red = provider.decide(ai_label="RED")
        second_red = provider.decide(signal_label="RED_LIGHT")
        green = provider.decide(ai_label="GREEN")

        self.assertEqual(Direction.STOP, first_red.action)
        self.assertEqual(Direction.STOP, second_red.action)
        self.assertEqual("traffic_light_hold", first_red.reason)
        self.assertEqual(Direction.RIGHT, green.action)
        self.assertEqual(1, provider.consumed)

    def test_separate_green_signal_and_ai_sign_can_be_used_together(self):
        provider = ScenarioDecisionProvider(make_scenario([
            {
                "id": "combined_ai",
                "allowed": ["LEFT", "RIGHT"],
                "action": "LEFT",
            }
        ]))

        result = provider.decide(
            ai_label="TURN_RIGHT",
            signal_label="GREEN_LIGHT",
        )

        self.assertEqual(Direction.RIGHT, result.action)
        self.assertEqual("TURN_RIGHT", result.sign_label)

    def test_disallowed_scripted_action_fails_closed(self):
        provider = ScenarioDecisionProvider(make_scenario([
            {
                "id": "unsafe_script",
                "allowed": ["LEFT"],
                "action": "RIGHT",
            }
        ]))

        result = provider.decide()

        self.assertEqual(Direction.STOP, result.action)
        self.assertIn("invalid_or_disallowed", result.reason)

    def test_invalid_scenario_and_intersection_fail_closed(self):
        invalid_root = ScenarioDecisionProvider({"intersections": "not-an-array"})
        invalid_entry = ScenarioDecisionProvider(make_scenario([
            {
                "id": "bad_allowed",
                "allowed": ["LEFT", "GREEN_ZONE"],
                "action": "LEFT",
            }
        ]))

        self.assertEqual(Direction.STOP, invalid_root.decide().action)
        self.assertTrue(invalid_root.decide().reason.startswith("invalid_scenario"))
        self.assertEqual(Direction.STOP, invalid_entry.decide().action)

    def test_scenario_exhaustion_stops_and_reset_rewinds(self):
        provider = ScenarioDecisionProvider(make_scenario([
            {
                "id": "only",
                "allowed": ["STRAIGHT"],
                "action": "STRAIGHT",
            }
        ]))

        self.assertEqual(Direction.STRAIGHT, provider.decide().action)
        exhausted = provider.decide()
        self.assertEqual(Direction.STOP, exhausted.action)
        self.assertEqual("scenario_exhausted", exhausted.reason)

        provider.reset()
        self.assertEqual(Direction.STRAIGHT, provider.decide().action)

    def test_example_json_is_loadable_and_all_actions_are_safe(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src",
            "smart_city",
            "v2",
            "scenario_example.json",
        )
        provider = ScenarioDecisionProvider(path)

        results = [provider.decide() for _ in range(provider.count)]

        self.assertEqual(5, len(results))
        for result in results[:-1]:
            self.assertNotEqual(Direction.STOP, result.action)
            self.assertIn(result.action, result.allowed)
        self.assertEqual(Direction.END, results[-1].action)
        self.assertEqual(0, provider.remaining)

        exhausted = provider.decide()
        self.assertEqual(Direction.STOP, exhausted.action)
        self.assertEqual("scenario_exhausted", exhausted.reason)


if __name__ == "__main__":
    unittest.main()
