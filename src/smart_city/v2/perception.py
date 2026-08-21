# -*- coding: utf-8 -*-
"""OpenCV perception for the Smart City course.

The module deliberately has no ROS or vehicle-control dependency.  It turns one
BGR camera frame into conservative observations that a state machine can use:

* a near and a far lane centre from white markers or island/course boundaries;
* a stop-line event only when several separate white bars form one row;
* green plus optional orange/red-boundary keep-out occupancy inside the
  projected driving corridor and a steering bias away from it.

All geometry is expressed as image ratios so the same code can be exercised on
small synthetic images and on the 640x480 camera stream.  Configuration may be
either a dictionary or an object with attributes.
"""

from __future__ import division

import math

import cv2
import numpy as np

class PerceptionResult(object):
    """Value object returned by :meth:`SmartCityPerception.analyze`.

    ``avoidance_bias`` follows the steering convention used by this module:
    positive means steer right (green is stronger on the left), negative means
    steer left.  A zero bias does not mean that the road is safe; callers must
    also inspect ``green_ahead_ratio`` and ``lane_confidence``.
    """

    __slots__ = (
        "frame",
        "white_mask",
        "green_mask",
        "lane_x_near",
        "lane_x_far",
        "lane_confidence",
        "lane_near_confidence",
        "lane_far_confidence",
        "lane_valid",
        "stop_line",
        "stop_line_score",
        "stop_line_y",
        "green_ahead_ratio",
        "green_left_ratio",
        "green_right_ratio",
        "green_danger",
        "avoidance_bias",
        "debug_frame",
    )

    def __init__(
        self,
        frame,
        white_mask,
        green_mask,
        lane_x_near,
        lane_x_far,
        lane_confidence,
        lane_near_confidence,
        lane_far_confidence,
        stop_line,
        stop_line_score,
        stop_line_y,
        green_ahead_ratio,
        green_left_ratio,
        green_right_ratio,
        green_danger,
        avoidance_bias,
        debug_frame,
    ):
        self.frame = frame
        self.white_mask = white_mask
        self.green_mask = green_mask
        self.lane_x_near = lane_x_near
        self.lane_x_far = lane_x_far
        self.lane_confidence = float(lane_confidence)
        self.lane_near_confidence = float(lane_near_confidence)
        self.lane_far_confidence = float(lane_far_confidence)
        self.lane_valid = lane_x_near is not None and lane_confidence > 0.0
        self.stop_line = bool(stop_line)
        self.stop_line_score = float(stop_line_score)
        self.stop_line_y = stop_line_y
        self.green_ahead_ratio = float(green_ahead_ratio)
        self.green_left_ratio = float(green_left_ratio)
        self.green_right_ratio = float(green_right_ratio)
        self.green_danger = bool(green_danger)
        self.avoidance_bias = float(avoidance_bias)
        self.debug_frame = debug_frame

    def as_dict(self):
        """Return a shallow dictionary without copying image arrays."""
        return {name: getattr(self, name) for name in self.__slots__}

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class SmartCityPerception(object):
    """Detect lane geometry, stop-line bars, and forbidden green regions."""

    def __init__(self, config=None):
        self.config = config

        # Resize only when both dimensions are explicitly available.  With an
        # empty config the input resolution is intentionally preserved.
        self.frame_width = self._cfg(
            ("frame_width", "image_width", "width", "FRAME_WIDTH", "WIDTH"),
            None,
        )
        self.frame_height = self._cfg(
            ("frame_height", "image_height", "height", "FRAME_HEIGHT", "HEIGHT"),
            None,
        )

        self.white_lower = self._hsv_value(
            self._cfg(("white_hsv_lower", "WHITE_HSV_LOWER"), (0, 0, 170))
        )
        self.white_upper = self._hsv_value(
            self._cfg(("white_hsv_upper", "WHITE_HSV_UPPER"), (179, 80, 255))
        )
        self.green_lower = self._hsv_value(
            self._cfg(("green_hsv_lower", "GREEN_HSV_LOWER"), (30, 45, 25))
        )
        self.green_upper = self._hsv_value(
            self._cfg(("green_hsv_upper", "GREEN_HSV_UPPER"), (95, 255, 255))
        )
        self.orange_keepout_enabled = bool(
            self._cfg(("orange_keepout_enabled",), False)
        )
        self.orange_lower_1 = self._hsv_value(
            self._cfg(("orange_hsv_lower_1",), (0, 80, 60))
        )
        self.orange_upper_1 = self._hsv_value(
            self._cfg(("orange_hsv_upper_1",), (25, 255, 255))
        )
        self.orange_lower_2 = self._hsv_value(
            self._cfg(("orange_hsv_lower_2",), (165, 80, 60))
        )
        self.orange_upper_2 = self._hsv_value(
            self._cfg(("orange_hsv_upper_2",), (179, 255, 255))
        )

        self.morph_kernel = max(
            1, int(self._cfg(("morph_kernel", "MORPH_KERNEL"), 3))
        )

        self.lane_roi_top = self._ratio(
            self._cfg(("lane_roi_top", "LANE_ROI_TOP"), 0.30), 0.0, 1.0
        )
        self.lane_roi_bottom = self._ratio(
            self._cfg(("lane_roi_bottom", "LANE_ROI_BOTTOM"), 0.98), 0.0, 1.0
        )
        self.scan_near_ratio = self._ratio(
            self._cfg(("scan_near_ratio", "SCAN_NEAR_RATIO"), 0.86), 0.0, 1.0
        )
        self.scan_far_ratio = self._ratio(
            self._cfg(("scan_far_ratio", "SCAN_FAR_RATIO"), 0.62), 0.0, 1.0
        )
        self.scan_half_height_ratio = self._ratio(
            self._cfg(
                ("lane_scan_half_height_ratio", "scan_half_height_ratio"), 0.09
            ),
            0.015,
            0.25,
        )
        self.lane_min_pixels = max(
            2, int(self._cfg(("lane_min_pixels", "LANE_MIN_PIXELS"), 12))
        )
        self.lane_min_width_ratio = self._ratio(
            self._cfg(("lane_min_width_ratio", "LANE_MIN_WIDTH_RATIO"), 0.18),
            0.05,
            0.90,
        )
        self.lane_max_width_ratio = self._ratio(
            self._cfg(("lane_max_width_ratio", "LANE_MAX_WIDTH_RATIO"), 0.78),
            self.lane_min_width_ratio + 0.01,
            0.98,
        )
        self.lane_pair_score_min = self._ratio(
            self._cfg(("lane_pair_score_min",), 0.32), 0.0, 1.0
        )
        default_lane_width_ratio = self._ratio(
            self._cfg(("lane_default_width_ratio",), 0.50),
            self.lane_min_width_ratio,
            self.lane_max_width_ratio,
        )
        self.expected_near_width_ratio = self._ratio(
            self._cfg(
                ("expected_lane_width_near_ratio",), default_lane_width_ratio
            ),
            self.lane_min_width_ratio,
            self.lane_max_width_ratio,
        )
        self.expected_far_width_ratio = self._ratio(
            self._cfg(
                ("expected_lane_width_far_ratio",),
                default_lane_width_ratio * 0.68,
            ),
            self.lane_min_width_ratio,
            self.lane_max_width_ratio,
        )

        self.stop_roi_top = self._ratio(
            self._cfg(("stop_roi_top", "STOP_ROI_TOP"), 0.30), 0.0, 1.0
        )
        self.stop_roi_bottom = self._ratio(
            self._cfg(("stop_roi_bottom", "STOP_ROI_BOTTOM"), 0.90), 0.0, 1.0
        )
        self.stop_min_components = max(
            3,
            int(
                self._cfg(
                    ("stop_min_components", "STOP_MIN_COMPONENTS"), 4
                )
            ),
        )
        self.stop_min_component_area = self._cfg(
            ("stop_min_component_area", "STOP_MIN_COMPONENT_AREA"), None
        )
        self.stop_max_component_area_ratio = self._ratio(
            self._cfg(("stop_max_component_area_ratio",), 0.04),
            0.001,
            0.25,
        )
        self.stop_cluster_y_px = self._cfg(
            ("stop_cluster_y_px", "STOP_CLUSTER_Y_PX"), None
        )
        self.stop_min_span_ratio = self._ratio(
            self._cfg(("stop_min_span_ratio", "STOP_MIN_SPAN_RATIO"), 0.25),
            0.08,
            0.95,
        )
        self.stop_score_threshold = self._ratio(
            self._cfg(("stop_score_threshold",), 0.60), 0.0, 1.0
        )

        self.green_roi_top = self._ratio(
            self._cfg(("green_roi_top", "GREEN_ROI_TOP"), 0.35), 0.0, 1.0
        )
        self.green_roi_bottom = self._ratio(
            self._cfg(("green_roi_bottom", "GREEN_ROI_BOTTOM"), 0.98), 0.0, 1.0
        )
        self.green_danger_ratio = max(
            1e-6,
            float(
                self._cfg(("green_danger_ratio", "GREEN_DANGER_RATIO"), 0.035)
            ),
        )
        self.green_bias_start_ratio = max(
            0.0,
            float(
                self._cfg(
                    ("green_bias_start_ratio",),
                    min(0.025, self.green_danger_ratio),
                )
            ),
        )

    def analyze(self, frame, previous_lane_x=None):
        """Analyze one BGR frame and return a :class:`PerceptionResult`.

        Missing or invalid frames raise ``ValueError``.  Missing lane evidence is
        represented by ``None`` centres and zero confidence; no image-centre
        fallback is emitted as a steering target.
        """
        image = self._prepare_frame(frame)
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        # ``green_mask`` keeps its public name for compatibility, but with the
        # official config it represents all forbidden green/orange/red cues.
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        if self.orange_keepout_enabled:
            orange_1 = cv2.inRange(hsv, self.orange_lower_1, self.orange_upper_1)
            orange_2 = cv2.inRange(hsv, self.orange_lower_2, self.orange_upper_2)
            green_mask = cv2.bitwise_or(
                green_mask, cv2.bitwise_or(orange_1, orange_2)
            )
        white_mask = self._clean_mask(white_mask, opening=True)
        green_mask = self._clean_mask(green_mask, opening=False)

        # The car starts close to the image centre and subsequently supplies
        # the last accepted lane centre.  Restricting a marker row to this ego
        # corridor prevents a zebra row on a perpendicular/adjacent road from
        # opening an intersection episode.
        anchor = self._valid_previous(previous_lane_x, width)
        if anchor is None:
            anchor = width * 0.5
        stop_info = self._detect_stop_line(white_mask, anchor)

        # A stop row is deliberately excluded from lane fitting.
        lane_mask = white_mask.copy()
        if stop_info["detected"]:
            labels = stop_info["labels_image"]
            for label_id in stop_info["label_ids"]:
                lane_mask[labels == label_id] = 0

        # Smart City lane geometry is intentionally independent from the Speed
        # Track BEV calibration.  The two scan bands provide both lateral and
        # heading error to the controller during normal following and recenter.
        near_paint = self._find_lane_pair(
            lane_mask,
            self.scan_near_ratio,
            anchor,
            self.expected_near_width_ratio,
        )
        near_keepout = self._find_keepout_corridor(
            green_mask,
            self.scan_near_ratio,
            anchor,
            self.expected_near_width_ratio,
        )
        near = self._prefer_painted_lane(near_paint, near_keepout)
        lane_x_near, near_confidence = near[0], near[1]

        # Associate the far pair with the current near corridor when possible;
        # this avoids jumping to the perpendicular road at an intersection.
        far_anchor = lane_x_near if lane_x_near is not None else anchor
        far_paint = self._find_lane_pair(
            lane_mask,
            self.scan_far_ratio,
            far_anchor,
            self.expected_far_width_ratio,
        )
        far_keepout = self._find_keepout_corridor(
            green_mask,
            self.scan_far_ratio,
            far_anchor,
            self.expected_far_width_ratio,
        )
        far = self._prefer_painted_lane(far_paint, far_keepout)
        lane_x_far, far_confidence = far[0], far[1]

        if lane_x_near is None:
            lane_confidence = 0.0
        elif lane_x_far is None:
            lane_confidence = 0.45 * near_confidence
        else:
            lane_confidence = 0.60 * near_confidence + 0.40 * far_confidence
        lane_confidence = self._clamp(lane_confidence, 0.0, 1.0)


        green_info = self._green_metrics(
            green_mask,
            near,
            far,
            lane_x_near,
            lane_x_far,
        )

        debug_frame = self._draw_debug(
            image,
            green_mask,
            near,
            far,
            lane_confidence,
            stop_info,
            green_info,
        )

        return PerceptionResult(
            frame=image,
            white_mask=white_mask,
            green_mask=green_mask,
            lane_x_near=lane_x_near,
            lane_x_far=lane_x_far,
            lane_confidence=lane_confidence,
            lane_near_confidence=near_confidence,
            lane_far_confidence=far_confidence,
            stop_line=stop_info["detected"],
            stop_line_score=stop_info["score"],
            stop_line_y=stop_info["y"],
            green_ahead_ratio=green_info["ahead"],
            green_left_ratio=green_info["left"],
            green_right_ratio=green_info["right"],
            green_danger=green_info["danger"],
            avoidance_bias=green_info["bias"],
            debug_frame=debug_frame,
        )

    def _detect_stop_line(self, mask, anchor):
        """Return one gate-proxy zebra row in the ego corridor.

        One row of short, aligned bars is a complete marker observation.  The
        detector never requires two rows.  If several valid longitudinal rows
        are simultaneously visible, it selects the farther/later row as the
        gate proxy while returning every row label for lane-mask suppression.
        """
        height, width = mask.shape
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, 8
        )
        min_area = self.stop_min_component_area
        if min_area is None:
            min_area = max(10, int(round(width * height * 0.00008)))
        else:
            min_area = max(1, int(min_area))

        tolerance = self.stop_cluster_y_px
        if tolerance is None:
            tolerance = max(3.0, height * 0.035)
        else:
            tolerance = max(1.0, float(tolerance))

        y_top = int(round(height * self.stop_roi_top))
        y_bottom = int(round(height * self.stop_roi_bottom))
        candidates = []
        for label_id in range(1, count):
            x, y, box_width, box_height, area = stats[label_id]
            center_x, center_y = centroids[label_id]
            if center_y < y_top or center_y > y_bottom:
                continue
            if area < min_area:
                continue
            if box_width < 2 or box_height < 2:
                continue
            # Large solid patches (floor glare, a page edge) are not one bar of
            # a segmented stop row.
            if (
                area > width * height * self.stop_max_component_area_ratio
                or box_width > width * 0.24
            ):
                continue
            candidates.append(
                {
                    "label": label_id,
                    "x": int(x),
                    "y": int(y),
                    "w": int(box_width),
                    "h": int(box_height),
                    "area": int(area),
                    "cx": float(center_x),
                    "cy": float(center_y),
                }
            )

        best = None
        best_score = 0.0
        valid_rows = {}
        # Each candidate is tried as the row centre.  This avoids transitive
        # grouping (y=10 with 15, 15 with 20) creating a slanted fake row.
        for seed in candidates:
            expected_width = self._expected_lane_width_at_y(
                seed["cy"], height, width
            )
            # Include paint at both edges while staying inside the road which
            # currently contains the car.  Adjacent/perpendicular road rows
            # therefore cannot become our stop marker merely because they are
            # visible in the same frame.
            corridor_half_width = max(width * 0.12, expected_width * 0.56)
            group = [
                item
                for item in candidates
                if abs(item["cy"] - seed["cy"]) <= tolerance
                and abs(item["cx"] - anchor) <= corridor_half_width
            ]
            group.sort(key=lambda item: item["cx"])
            group = self._deduplicate_row_components(group, width)
            if len(group) < self.stop_min_components:
                continue

            left = min(item["x"] for item in group)
            right = max(item["x"] + item["w"] for item in group)
            span = max(0, right - left)
            if span < width * self.stop_min_span_ratio:
                continue
            # A valid zebra row crosses the projected car centre.  A partial
            # row wholly on one side is treated as side-road evidence only.
            if left > anchor or right < anchor:
                continue

            centers_y = np.asarray([item["cy"] for item in group], dtype=np.float32)
            alignment = 1.0 - min(1.0, float(np.std(centers_y)) / tolerance)
            count_score = min(1.0, len(group) / float(self.stop_min_components))
            span_score = min(
                1.0, span / float(max(1.0, width * self.stop_min_span_ratio))
            )
            coverage = sum(item["w"] for item in group) / float(max(1, span))
            coverage_score = min(1.0, coverage / 0.22)
            score = (
                0.35 * count_score
                + 0.30 * span_score
                + 0.25 * alignment
                + 0.10 * coverage_score
            )
            if score >= self.stop_score_threshold:
                signature = tuple(sorted(item["label"] for item in group))
                previous = valid_rows.get(signature)
                if previous is None or score > previous[0]:
                    # Detection is deliberately restricted to the ego
                    # corridor, but once a crossing row is valid, remove all
                    # same-row bar components from lane fitting.  Otherwise
                    # the outer zebra bars can be paired as two longitudinal
                    # boundaries and fabricate a high-confidence lane.
                    mask_group = [
                        item for item in candidates
                        if abs(item["cy"] - seed["cy"]) <= tolerance
                    ]
                    valid_rows[signature] = (score, mask_group)
            if score > best_score or (
                abs(score - best_score) <= 1e-9
                and best is not None
                and seed["cy"] > np.mean([item["cy"] for item in best])
            ):
                best, best_score = group, score

        if valid_rows:
            # When two longitudinal marker rows are visible, the row farther
            # from the camera is the later row along the approach and is the
            # better proxy for the actual junction gate.  With only one row it
            # is selected unchanged; no row-count assumption is required.
            best_score, best = min(
                valid_rows.values(),
                key=lambda item: float(np.mean([
                    component["cy"] for component in item[1]
                ])),
            )
        detected = best is not None and best_score >= self.stop_score_threshold
        if not detected:
            best = []
            best_score = 0.0
        stop_y = None
        if best:
            stop_y = int(round(float(np.mean([item["cy"] for item in best]))))
        marker_label_ids = sorted({
            item["label"]
            for _score, row in valid_rows.values()
            for item in row
        })
        return {
            "detected": detected,
            "score": self._clamp(best_score, 0.0, 1.0),
            "y": stop_y,
            "components": best,
            # Remove every valid zebra row from lane fitting.  When the camera
            # sees two markers 20--45 cm apart, leaving the second row in the
            # white mask could make its bars look like lane boundaries.
            "label_ids": marker_label_ids,
            "labels_image": labels,
        }

    def _expected_lane_width_at_y(self, y_value, image_height, image_width):
        """Interpolate calibrated lane width between far and near scans."""
        far_y = image_height * self.scan_far_ratio
        near_y = image_height * self.scan_near_ratio
        if abs(near_y - far_y) <= 1e-9:
            progress = 1.0
        else:
            progress = self._clamp(
                (float(y_value) - far_y) / float(near_y - far_y),
                0.0,
                1.0,
            )
        width_ratio = (
            self.expected_far_width_ratio
            + progress
            * (self.expected_near_width_ratio - self.expected_far_width_ratio)
        )
        return image_width * width_ratio

    @staticmethod
    def _deduplicate_row_components(group, image_width):
        if not group:
            return []
        minimum_separation = max(2.0, image_width * 0.008)
        result = []
        for item in group:
            if result and item["cx"] - result[-1]["cx"] < minimum_separation:
                if item["area"] > result[-1]["area"]:
                    result[-1] = item
            else:
                result.append(item)
        return result

    def _find_lane_pair(self, mask, scan_ratio, anchor, expected_width_ratio):
        height, width = mask.shape
        roi_top = int(round(height * self.lane_roi_top))
        roi_bottom = int(round(height * self.lane_roi_bottom))
        scan_y = int(round(height * scan_ratio))
        half_height = max(3, int(round(height * self.scan_half_height_ratio)))
        y0 = max(roi_top, scan_y - half_height)
        y1 = min(roi_bottom, scan_y + half_height + 1)
        if y1 <= y0:
            return None, 0.0, None, None, scan_y

        band = mask[y0:y1]
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(band, 8)
        items = []
        for label_id in range(1, count):
            x, _y, box_width, box_height, area = stats[label_id]
            if area < self.lane_min_pixels:
                continue
            if box_width < 2 or box_height < 2:
                continue
            if box_width > width * 0.22 or box_height > height * 0.30:
                continue
            items.append(
                {
                    "x": float(centroids[label_id][0]),
                    "area": float(area),
                    "width": int(box_width),
                }
            )

        clusters = self._cluster_lane_items(items, width)
        if len(clusters) < 2:
            return None, 0.0, None, None, scan_y

        min_width = width * self.lane_min_width_ratio
        max_width = width * self.lane_max_width_ratio
        expected_width = width * expected_width_ratio
        anchor_tolerance = width * float(
            self._cfg(("lane_anchor_tolerance_ratio",), 0.10)
        )

        best = None
        best_score = -1.0
        for left_index in range(len(clusters) - 1):
            for right_index in range(left_index + 1, len(clusters)):
                left = clusters[left_index]
                right = clusters[right_index]
                lane_width = right["x"] - left["x"]
                if lane_width < min_width or lane_width > max_width:
                    continue
                # The current/previous car centre should be enclosed by the
                # selected lane.  The tolerance admits ordinary curved lanes.
                if left["x"] > anchor + anchor_tolerance:
                    continue
                if right["x"] < anchor - anchor_tolerance:
                    continue

                center = 0.5 * (left["x"] + right["x"])
                center_distance = abs(center - anchor)
                center_score = max(0.0, 1.0 - center_distance / (width * 0.35))
                width_score = max(
                    0.0,
                    1.0
                    - abs(lane_width - expected_width)
                    / max(1.0, max_width - min_width),
                )
                minimum_support = min(left["support"], right["support"])
                support_score = min(
                    1.0, minimum_support / float(self.lane_min_pixels * 2)
                )
                balance_score = minimum_support / max(
                    left["support"], right["support"]
                )
                score = (
                    0.38 * support_score
                    + 0.28 * center_score
                    + 0.22 * width_score
                    + 0.12 * balance_score
                )
                if score > best_score:
                    best_score = score
                    best = (center, left["x"], right["x"])

        if best is None or best_score < self.lane_pair_score_min:
            return None, 0.0, None, None, scan_y
        return (
            float(best[0]),
            self._clamp(best_score, 0.0, 1.0),
            float(best[1]),
            float(best[2]),
            scan_y,
        )

    def _find_keepout_corridor(
        self, keepout_mask, scan_ratio, anchor, expected_width_ratio
    ):
        """Find the free corridor between the nearest forbidden boundaries.

        Filled green islands and their red/orange outlines are already folded
        into ``keepout_mask``.  The nearest occupied column on either side of
        the tracked car centre therefore provides a useful lane fallback on
        long road sections which have no white paint.
        """
        height, width = keepout_mask.shape
        roi_top = int(round(height * self.lane_roi_top))
        roi_bottom = int(round(height * self.lane_roi_bottom))
        scan_y = int(round(height * scan_ratio))
        half_height = max(3, int(round(height * self.scan_half_height_ratio)))
        y0 = max(roi_top, scan_y - half_height)
        y1 = min(roi_bottom, scan_y + half_height + 1)
        empty = (None, 0.0, None, None, scan_y)
        if y1 <= y0:
            return empty

        band = keepout_mask[y0:y1] > 0
        column_support = np.count_nonzero(band, axis=0)
        # A hard island/course boundary must persist through a meaningful
        # portion of the longitudinal scan band.  The former 6% threshold let
        # a few isolated coloured pixels at both scan rows fabricate a
        # high-confidence lane and arm the vehicle on an otherwise blank road.
        minimum_support = max(3, int(round((y1 - y0) * 0.30)))
        occupied = column_support >= minimum_support

        anchor_index = int(round(anchor))
        anchor_index = max(0, min(width - 1, anchor_index))
        if occupied[anchor_index]:
            return empty

        left_indices = np.flatnonzero(occupied[:anchor_index])
        right_indices = np.flatnonzero(occupied[anchor_index + 1 :])
        if left_indices.size == 0 or right_indices.size == 0:
            return empty

        left_x = int(left_indices[-1])
        right_x = int(anchor_index + 1 + right_indices[0])
        lane_width = right_x - left_x
        min_width = width * self.lane_min_width_ratio
        max_width = width * self.lane_max_width_ratio
        if lane_width < min_width or lane_width > max_width:
            return empty

        expected_width = width * expected_width_ratio
        width_score = max(
            0.0,
            1.0
            - abs(lane_width - expected_width)
            / max(1.0, max_width - min_width),
        )
        support_window = max(2, int(round(width * 0.012)))
        left_support = np.max(
            column_support[max(0, left_x - support_window + 1) : left_x + 1]
        )
        right_support = np.max(
            column_support[right_x : min(width, right_x + support_window)]
        )
        support_score = min(
            1.0,
            min(left_support, right_support) / float(max(1, y1 - y0)),
        )
        confidence = self._clamp(
            0.25 + 0.45 * width_score + 0.30 * support_score,
            0.0,
            1.0,
        )
        return (
            0.5 * (left_x + right_x),
            confidence,
            float(left_x),
            float(right_x),
            scan_y,
        )

    @staticmethod
    def _prefer_painted_lane(painted, keepout):
        """Prefer explicit paint; use hard island/course edges as fallback."""
        if painted[0] is not None:
            return painted
        return keepout

    def _cluster_lane_items(self, items, image_width):
        if not items:
            return []
        tolerance = float(
            self._cfg(("lane_cluster_x_px",), max(4.0, image_width * 0.055))
        )
        clusters = []
        for item in sorted(items, key=lambda value: value["x"]):
            nearest = None
            nearest_distance = None
            for cluster in clusters:
                distance = abs(item["x"] - cluster["x"])
                if distance <= tolerance and (
                    nearest_distance is None or distance < nearest_distance
                ):
                    nearest = cluster
                    nearest_distance = distance
            if nearest is None:
                clusters.append(
                    {
                        "x": item["x"],
                        "support": item["area"],
                        "count": 1,
                    }
                )
            else:
                total = nearest["support"] + item["area"]
                nearest["x"] = (
                    nearest["x"] * nearest["support"]
                    + item["x"] * item["area"]
                ) / total
                nearest["support"] = total
                nearest["count"] += 1
        return sorted(clusters, key=lambda value: value["x"])

    def _green_metrics(
        self, green_mask, near, far, lane_x_near, lane_x_far
    ):
        height, width = green_mask.shape
        y0 = max(0, int(round(height * self.green_roi_top)))
        y1 = min(height, int(round(height * self.green_roi_bottom)))
        if y1 <= y0:
            return {"ahead": 0.0, "left": 0.0, "right": 0.0,
                    "danger": False, "bias": 0.0, "corridor": None}

        near_center = lane_x_near if lane_x_near is not None else width * 0.5
        far_center = lane_x_far if lane_x_far is not None else near_center

        near_half = self._lane_half_width(near, width)
        far_half = self._lane_half_width(far, width)
        rows = np.arange(y0, y1, dtype=np.float32)
        denominator = float(max(1, y1 - y0 - 1))
        progress = (rows - y0) / denominator
        centers = far_center + progress * (near_center - far_center)
        half_widths = far_half + progress * (near_half - far_half)

        x_values = np.arange(width, dtype=np.float32)[None, :]
        centers_2d = centers[:, None]
        half_widths_2d = half_widths[:, None]
        corridor = np.abs(x_values - centers_2d) <= half_widths_2d
        left_region = corridor & (x_values < centers_2d)
        right_region = corridor & (x_values >= centers_2d)
        green = green_mask[y0:y1] > 0

        ahead = self._masked_ratio(green, corridor)
        left = self._masked_ratio(green, left_region)
        right = self._masked_ratio(green, right_region)
        danger = ahead >= self.green_danger_ratio

        total_side = left + right
        strongest_side = max(left, right)
        if total_side <= 1e-9 or strongest_side <= self.green_bias_start_ratio:
            bias = 0.0
        else:
            direction = (left - right) / total_side
            strength_range = max(
                1e-6, self.green_danger_ratio - self.green_bias_start_ratio
            )
            strength = min(
                1.0,
                (strongest_side - self.green_bias_start_ratio) / strength_range,
            )
            bias = self._clamp(direction * strength, -1.0, 1.0)

        polygon = np.asarray(
            [
                [int(round(far_center - far_half)), y0],
                [int(round(far_center + far_half)), y0],
                [int(round(near_center + near_half)), y1 - 1],
                [int(round(near_center - near_half)), y1 - 1],
            ],
            dtype=np.int32,
        )
        return {
            "ahead": ahead,
            "left": left,
            "right": right,
            "danger": danger,
            "bias": bias,
            "corridor": polygon,
        }

    @staticmethod
    def _lane_half_width(lane, image_width):
        if lane[2] is not None and lane[3] is not None:
            # Stay slightly inside the detected paint.  Green beyond a white
            # road boundary should not create unnecessary evasive steering.
            return max(image_width * 0.10, (lane[3] - lane[2]) * 0.45)
        return image_width * 0.23

    @staticmethod
    def _masked_ratio(values, region):
        pixels = int(np.count_nonzero(region))
        if pixels == 0:
            return 0.0
        return float(np.count_nonzero(values & region)) / float(pixels)

    def _draw_debug(
        self,
        image,
        green_mask,
        near,
        far,
        lane_confidence,
        stop_info,
        green_info,
    ):
        debug = image.copy()
        height, width = debug.shape[:2]

        # Transparent green tint keeps the original road texture visible.
        green_pixels = green_mask > 0
        if np.any(green_pixels):
            tint = debug.copy()
            tint[green_pixels] = (0, 190, 0)
            debug = cv2.addWeighted(debug, 0.72, tint, 0.28, 0.0)

        corridor = green_info["corridor"]
        if corridor is not None:
            cv2.polylines(debug, [corridor], True, (0, 165, 255), 1)

        for lane, color in ((far, (255, 200, 0)), (near, (0, 255, 255))):
            center, _confidence, left, right, scan_y = lane
            cv2.line(debug, (0, scan_y), (width - 1, scan_y), (80, 80, 80), 1)
            if center is not None:
                if left is not None:
                    cv2.circle(debug, (int(round(left)), scan_y), 4, color, -1)
                if right is not None:
                    cv2.circle(debug, (int(round(right)), scan_y), 4, color, -1)
                cv2.circle(debug, (int(round(center)), scan_y), 5, (0, 0, 255), -1)

        # Draw green synthetic line connecting far to near center
        if far[0] is not None and near[0] is not None:
            cv2.line(debug, (int(round(far[0])), far[4]), (int(round(near[0])), near[4]), (0, 255, 0), 2)

        for component in stop_info["components"]:
            cv2.rectangle(
                debug,
                (component["x"], component["y"]),
                (
                    component["x"] + component["w"] - 1,
                    component["y"] + component["h"] - 1,
                ),
                (0, 0, 255),
                2,
            )

        status = "lane {:.2f}".format(lane_confidence)
        cv2.putText(
            debug,
            status,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            "stop {} {:.2f} green {:.3f}".format(
                int(stop_info["detected"]),
                stop_info["score"],
                green_info["ahead"],
            ),
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return debug

    def _prepare_frame(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        if frame.ndim == 2:
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            image = frame
        elif frame.ndim == 3 and frame.shape[2] == 4:
            image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError("frame must be grayscale, BGR, or BGRA")

        if self.frame_width is not None and self.frame_height is not None:
            target = (int(self.frame_width), int(self.frame_height))
            if target[0] <= 0 or target[1] <= 0:
                raise ValueError("configured frame dimensions must be positive")
            if image.shape[1] != target[0] or image.shape[0] != target[1]:
                image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image)

    def _clean_mask(self, mask, opening):
        kernel_size = self.morph_kernel
        if kernel_size <= 1:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        if opening:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _cfg(self, names, default):
        if isinstance(names, str):
            names = (names,)
        if self.config is None:
            return default
        for name in names:
            if isinstance(self.config, dict) and name in self.config:
                return self.config[name]
            if hasattr(self.config, name):
                return getattr(self.config, name)
        return default

    @staticmethod
    def _hsv_value(value):
        array = np.asarray(value, dtype=np.int32).reshape(-1).copy()
        if array.size != 3:
            raise ValueError("HSV threshold must contain exactly three values")
        array[0] = int(max(0, min(179, array[0])))
        array[1:] = np.clip(array[1:], 0, 255)
        return array.astype(np.uint8)

    @staticmethod
    def _valid_previous(previous_lane_x, image_width):
        if previous_lane_x is None:
            return None
        try:
            value = float(previous_lane_x)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0.0 or value >= image_width:
            return None
        return value

    @staticmethod
    def _ratio(value, lower, upper):
        return SmartCityPerception._clamp(float(value), lower, upper)

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, float(value)))
