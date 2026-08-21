# -*- coding: utf-8 -*-
"""Synthetic unit tests for Smart City V2 perception (no ROS/hardware)."""

from __future__ import division

import unittest
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.smart_city.v2.perception import PerceptionResult, SmartCityPerception


class SyntheticConfig(object):
    frame_width = 320
    frame_height = 240
    white_hsv_lower = (0, 0, 180)
    white_hsv_upper = (179, 45, 255)
    green_hsv_lower = (35, 80, 30)
    green_hsv_upper = (90, 255, 255)
    morph_kernel = 3

    lane_roi_top = 0.32
    lane_roi_bottom = 0.98
    scan_near_ratio = 0.84
    scan_far_ratio = 0.56
    lane_scan_half_height_ratio = 0.10
    lane_min_pixels = 24
    lane_min_width_ratio = 0.20
    lane_max_width_ratio = 0.72

    stop_roi_top = 0.35
    stop_roi_bottom = 0.90
    stop_min_components = 4
    stop_min_component_area = 35
    stop_cluster_y_px = 7
    stop_min_span_ratio = 0.25

    green_roi_top = 0.38
    green_roi_bottom = 0.97
    green_danger_ratio = 0.025


def blank_road():
    return np.full((240, 320, 3), (28, 28, 28), dtype=np.uint8)


def draw_lane(frame, center_x=160, omit_y_range=None):
    """Draw two dashed boundaries converging towards the horizon."""
    dash_centers_y = (92, 122, 154, 188, 220)
    for center_y in dash_centers_y:
        if omit_y_range is not None:
            if omit_y_range[0] <= center_y <= omit_y_range[1]:
                continue
        perspective = (center_y - 92) / float(220 - 92)
        half_width = 43 + perspective * 42
        marker_width = int(round(9 + perspective * 5))
        marker_height = int(round(13 + perspective * 6))
        for x_center in (center_x - half_width, center_x + half_width):
            x0 = int(round(x_center - marker_width / 2.0))
            y0 = int(round(center_y - marker_height / 2.0))
            cv2.rectangle(
                frame,
                (x0, y0),
                (x0 + marker_width, y0 + marker_height),
                (245, 245, 245),
                -1,
            )
    return frame


def draw_lane_with_heading(frame, far_center_x=145, near_center_x=180):
    """Draw a corridor whose near/far centres encode a heading error."""
    dash_centers_y = (92, 122, 154, 188, 220)
    for center_y in dash_centers_y:
        progress = (center_y - 92) / float(220 - 92)
        center_x = far_center_x + progress * (near_center_x - far_center_x)
        half_width = 43 + progress * 42
        marker_width = int(round(9 + progress * 5))
        marker_height = int(round(13 + progress * 6))
        for x_center in (center_x - half_width, center_x + half_width):
            x0 = int(round(x_center - marker_width / 2.0))
            y0 = int(round(center_y - marker_height / 2.0))
            cv2.rectangle(
                frame,
                (x0, y0),
                (x0 + marker_width, y0 + marker_height),
                (245, 245, 245),
                -1,
            )
    return frame


class SmartCityPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.perception = SmartCityPerception(SyntheticConfig())

    def test_detects_near_and_far_lane_centres(self):
        result = self.perception.analyze(draw_lane(blank_road()))

        self.assertIsInstance(result, PerceptionResult)
        self.assertTrue(result.lane_valid)
        self.assertIsNotNone(result.lane_x_near)
        self.assertIsNotNone(result.lane_x_far)
        self.assertAlmostEqual(result.lane_x_near, 160.0, delta=5.0)
        self.assertAlmostEqual(result.lane_x_far, 160.0, delta=5.0)
        self.assertGreater(result.lane_confidence, 0.55)
        self.assertEqual(result.white_mask.shape, (240, 320))
        self.assertEqual(result.debug_frame.shape, (240, 320, 3))
        self.assertIs(result["white_mask"], result.white_mask)

    def test_previous_lane_x_associates_a_shifted_lane(self):
        result = self.perception.analyze(
            draw_lane(blank_road(), center_x=126), previous_lane_x=126
        )

        self.assertTrue(result.lane_valid)
        self.assertAlmostEqual(result.lane_x_near, 126.0, delta=5.0)
        self.assertAlmostEqual(result.lane_x_far, 126.0, delta=5.0)

    def test_near_and_far_centres_preserve_heading_for_recenter(self):
        result = self.perception.analyze(
            draw_lane_with_heading(blank_road()), previous_lane_x=180
        )

        self.assertTrue(result.lane_valid)
        self.assertIsNotNone(result.lane_x_far)
        self.assertGreater(result.lane_x_near, result.lane_x_far + 12.0)
        self.assertGreater(result.lane_near_confidence, 0.0)
        self.assertGreater(result.lane_far_confidence, 0.0)

    def test_green_island_boundaries_form_lane_when_white_paint_is_absent(self):
        frame = blank_road()
        cv2.rectangle(frame, (0, 70), (80, 239), (0, 150, 0), -1)
        cv2.rectangle(frame, (240, 70), (319, 239), (0, 150, 0), -1)

        result = self.perception.analyze(frame, previous_lane_x=160)

        self.assertTrue(result.lane_valid)
        self.assertAlmostEqual(result.lane_x_near, 160.0, delta=3.0)
        self.assertAlmostEqual(result.lane_x_far, 160.0, delta=3.0)

    def test_sparse_green_noise_cannot_fabricate_a_lane(self):
        frame = blank_road()
        # Small blobs at both near/far scan rows used to satisfy the very low
        # keep-out column-support threshold and create a centred fake lane.
        for y in (int(240 * 0.56), int(240 * 0.84)):
            cv2.rectangle(frame, (75, y - 3), (81, y + 3), (0, 150, 0), -1)
            cv2.rectangle(frame, (239, y - 3), (245, y + 3), (0, 150, 0), -1)

        result = self.perception.analyze(frame, previous_lane_x=160)

        self.assertFalse(result.lane_valid)
        self.assertIsNone(result.lane_x_near)
        self.assertIsNone(result.lane_x_far)

    def test_missing_lane_is_fail_closed(self):
        result = self.perception.analyze(blank_road(), previous_lane_x=150)

        self.assertFalse(result.lane_valid)
        self.assertIsNone(result.lane_x_near)
        self.assertIsNone(result.lane_x_far)
        self.assertEqual(result.lane_confidence, 0.0)
        self.assertFalse(result.stop_line)

    def test_one_zebra_row_inside_ego_corridor_is_a_complete_marker(self):
        frame = draw_lane(blank_road(), omit_y_range=(140, 175))
        for x in (72, 106, 140, 174, 208, 242):
            cv2.rectangle(frame, (x, 146), (x + 11, 166), (250, 250, 250), -1)

        result = self.perception.analyze(frame)

        self.assertTrue(result.stop_line)
        self.assertGreaterEqual(result.stop_line_score, 0.60)
        self.assertAlmostEqual(result.stop_line_y, 156, delta=2)

    def test_two_visible_zebra_rows_are_both_removed_from_lane_evidence(self):
        mask = np.zeros((240, 320), dtype=np.uint8)
        frame = blank_road()
        for y in (118, 176):
            for x in (72, 106, 140, 174, 208, 242):
                cv2.rectangle(mask, (x, y), (x + 11, y + 16), 255, -1)
                cv2.rectangle(
                    frame, (x, y), (x + 11, y + 16), (250, 250, 250), -1
                )

        marker = self.perception._detect_stop_line(mask, anchor=160.0)
        result = self.perception.analyze(frame, previous_lane_x=160)

        self.assertTrue(marker["detected"])
        # Four central bars per row prove the marker crosses the ego corridor;
        # all six same-row bars must then be removed from lane evidence.
        self.assertEqual(12, len(marker["label_ids"]))
        # With two valid rows the farther/later row is the gate proxy; with one
        # row the exact same detector still uses that row.
        self.assertAlmostEqual(126, marker["y"], delta=2)
        self.assertTrue(result.stop_line)
        self.assertFalse(result.lane_valid)

    def test_zebra_row_outside_ego_corridor_is_ignored(self):
        frame = blank_road()
        for x in (4, 25, 46, 67, 88):
            cv2.rectangle(frame, (x, 146), (x + 11, 166), (250, 250, 250), -1)

        result = self.perception.analyze(frame, previous_lane_x=160)

        self.assertFalse(result.stop_line)
        self.assertEqual(result.stop_line_score, 0.0)

    def test_one_longitudinal_dashed_marker_is_not_a_stop_line(self):
        frame = blank_road()
        for center_y in (90, 125, 160, 195, 225):
            cv2.rectangle(
                frame,
                (151, center_y - 7),
                (169, center_y + 7),
                (250, 250, 250),
                -1,
            )

        result = self.perception.analyze(frame)

        self.assertFalse(result.stop_line)
        self.assertEqual(result.stop_line_score, 0.0)

    def test_too_few_aligned_bars_are_not_a_stop_line(self):
        frame = blank_road()
        for x in (92, 148, 204):
            cv2.rectangle(frame, (x, 145), (x + 13, 166), (250, 250, 250), -1)

        result = self.perception.analyze(frame)

        self.assertFalse(result.stop_line)
        self.assertEqual(result.stop_line_score, 0.0)

    def test_green_intrusion_reports_danger_and_biases_away(self):
        frame = draw_lane(blank_road())
        # Green enters the left half of the projected driving corridor.
        cv2.rectangle(frame, (98, 150), (151, 231), (0, 150, 0), -1)

        result = self.perception.analyze(frame)

        self.assertGreater(result.green_ahead_ratio, SyntheticConfig.green_danger_ratio)
        self.assertTrue(result.green_danger)
        self.assertGreater(result.green_left_ratio, result.green_right_ratio)
        self.assertGreater(result.avoidance_bias, 0.25)

    def test_orange_boundary_is_an_optional_keepout_cue(self):
        class OrangeConfig(SyntheticConfig):
            orange_keepout_enabled = True
            orange_hsv_lower_1 = (0, 80, 60)
            orange_hsv_upper_1 = (25, 255, 255)
            orange_hsv_lower_2 = (165, 80, 60)
            orange_hsv_upper_2 = (179, 255, 255)

        frame = draw_lane(blank_road())
        cv2.rectangle(frame, (98, 150), (151, 231), (0, 120, 255), -1)
        result = SmartCityPerception(OrangeConfig()).analyze(frame)

        self.assertTrue(result.green_danger)
        self.assertGreater(result.green_left_ratio, result.green_right_ratio)
        self.assertGreater(np.count_nonzero(result.green_mask), 0)

    def test_red_boundary_is_an_optional_keepout_cue(self):
        class RedBoundaryConfig(SyntheticConfig):
            orange_keepout_enabled = True
            orange_hsv_lower_1 = (0, 80, 60)
            orange_hsv_upper_1 = (25, 255, 255)
            orange_hsv_lower_2 = (165, 80, 60)
            orange_hsv_upper_2 = (179, 255, 255)

        frame = draw_lane(blank_road())
        cv2.rectangle(frame, (98, 150), (151, 231), (0, 0, 220), -1)
        result = SmartCityPerception(RedBoundaryConfig()).analyze(frame)

        self.assertTrue(result.green_danger)
        self.assertGreater(result.green_left_ratio, result.green_right_ratio)
        self.assertGreater(np.count_nonzero(result.green_mask), 0)

    def test_red_course_edges_form_lane_when_white_paint_is_absent(self):
        class RedEdgeConfig(SyntheticConfig):
            orange_keepout_enabled = True
            orange_hsv_lower_1 = (0, 80, 60)
            orange_hsv_upper_1 = (25, 255, 255)
            orange_hsv_lower_2 = (165, 80, 60)
            orange_hsv_upper_2 = (179, 255, 255)

        frame = blank_road()
        cv2.rectangle(frame, (76, 70), (82, 239), (0, 0, 220), -1)
        cv2.rectangle(frame, (238, 70), (244, 239), (0, 0, 220), -1)
        result = SmartCityPerception(RedEdgeConfig()).analyze(
            frame, previous_lane_x=160
        )

        self.assertTrue(result.lane_valid)
        self.assertAlmostEqual(result.lane_x_near, 160.0, delta=4.0)
        self.assertAlmostEqual(result.lane_x_far, 160.0, delta=4.0)

    def test_accepts_dictionary_config_and_rejects_invalid_frame(self):
        perception = SmartCityPerception(
            {
                "white_hsv_lower": (0, 0, 180),
                "white_hsv_upper": (179, 45, 255),
                "morph_kernel": 1,
            }
        )
        result = perception.analyze(np.full((80, 100, 3), 20, dtype=np.uint8))
        self.assertEqual(result.frame.shape, (80, 100, 3))
        with self.assertRaises(ValueError):
            perception.analyze(None)


if __name__ == "__main__":
    unittest.main()
