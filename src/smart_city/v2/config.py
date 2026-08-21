# -*- coding: utf-8 -*-
"""Central configuration for the Smart City V2 controller.

The values below are deliberately conservative.  They are starting points for
bench/offline tests, not a promise that a real car is calibrated.  Keep this
module free of ROS imports so perception and control can be unit-tested on a
laptop.
"""

import math


class SmartCityConfig(object):
    """Mutable Python 3.6 compatible configuration object."""

    def __init__(self):
        # Camera / colour segmentation.
        self.frame_width = 640
        self.frame_height = 480
        self.white_hsv_lower = (0, 0, 165)
        self.white_hsv_upper = (179, 78, 255)
        self.green_hsv_lower = (32, 45, 30)
        self.green_hsv_upper = (100, 255, 255)
        # The official mat outlines forbidden islands/course edges in orange
        # or red.  Fold those pixels into the keep-out mask as a second cue.
        self.orange_keepout_enabled = True
        self.orange_hsv_lower_1 = (0, 80, 60)
        self.orange_hsv_upper_1 = (25, 255, 255)
        self.orange_hsv_lower_2 = (165, 80, 60)
        self.orange_hsv_upper_2 = (179, 255, 255)
        self.morph_kernel = 3

        # Road/lane measurements. Ratios are relative to image dimensions.
        self.lane_roi_top = 0.35
        self.lane_roi_bottom = 0.96
        self.scan_near_ratio = 0.86
        self.scan_far_ratio = 0.62
        self.lane_min_pixels = 7
        self.lane_min_width_ratio = 0.22
        self.lane_max_width_ratio = 0.92
        self.lane_default_width_ratio = 0.48
        self.lane_min_confidence = 0.24

        # The marker before a junction consists of several short white bars.
        self.stop_roi_top = 0.45
        self.stop_roi_bottom = 0.88
        self.stop_min_components = 4
        self.stop_min_component_area = 22
        self.stop_max_component_area_ratio = 0.035
        self.stop_cluster_y_px = 24
        self.stop_min_span_ratio = 0.38
        self.stop_confirm_frames = 3
        self.stop_y_backtrack_tolerance_ratio = 0.06
        self.stop_approach_y_ratio = 0.52
        self.stop_close_y_ratio = 0.70

        # Green islands are forbidden.  Ratios are mask coverage within each
        # probe region, not a metric distance.
        self.green_roi_top = 0.48
        self.green_roi_bottom = 0.98
        self.green_danger_ratio = 0.055
        self.green_bias_start_ratio = 0.025
        self.turn_green_hard_stop_ratio = 0.18

        # Closed-loop lane following (normalised image error -> steering).
        self.lane_kp = 0.92
        self.lane_kd = 0.10
        self.heading_gain = 0.42
        self.green_avoidance_gain = 0.60
        self.max_lane_steering = 0.72
        self.steering_slew_per_second = 2.8

        # Throttle values are intentionally low for initial calibration.
        self.cruise_throttle = 0.15
        self.approach_throttle = 0.14
        self.turn_throttle = 0.12
        self.reacquire_throttle = 0.09
        self.straight_throttle = 0.12

        # Junction FSM timing. All time values are seconds (never frame counts).
        self.stop_hold_seconds = 0.55
        self.red_light_timeout_seconds = 20.0
        self.nudge_left_seconds = 0.34
        self.nudge_right_seconds = 0.30
        self.nudge_max_seconds = 1.50
        self.nudge_marker_clear_frames = 3
        self.turn_min_seconds = 0.72
        self.turn_nominal_seconds = 1.15
        self.turn_max_seconds = 1.80
        self.straight_cross_seconds = 1.05
        self.reacquire_timeout_seconds = 2.20
        self.reacquire_stable_frames = 5
        self.turn_lane_confirm_frames = 3
        self.reacquire_center_error_ratio = 0.16
        self.reacquire_heading_error_ratio = 0.12
        self.intersection_cooldown_seconds = 1.80
        self.exit_clear_frames = 8
        self.exit_lockout_max_seconds = 3.80
        self.turn_steering_left = -0.76
        self.turn_steering_right = 0.76

        # Fail-closed safety/watchdogs.
        self.camera_timeout_seconds = 0.30
        self.actuator_watchdog_seconds = 0.20
        self.initial_lane_stable_frames = 5
        self.sensor_acquire_timeout_seconds = 2.0
        self.arm_request_ttl_seconds = 0.75
        self.lane_loss_stop_seconds = 0.12
        self.lane_loss_estop_seconds = 0.55
        self.green_danger_confirm_frames = 2
        self.lidar_stop_distance_m = 0.25
        self.lidar_timeout_seconds = 0.30
        self.lidar_guard_half_angle_deg = 35.0
        self.lidar_yaw_offset_deg = 0.0
        self.loop_hz = 20.0
        self.ai_min_confidence = 0.60
        self.ai_confirm_frames = 3
        # A delayed GREEN is unsafe when the physical light may already have
        # changed.  Keep live semantic freshness close to the camera watchdog.
        self.semantic_ttl_seconds = 0.35

        # Explicitly isolated camera-only hardware bench mode.  This is not a
        # live-course calibration: the runner accepts it only with the
        # dedicated --bench-camera-only flag and enforces a short wall-clock
        # runtime plus a fixed two-turn scenario.
        self.bench_only = False
        self.bench_max_runtime_seconds = 25.0

        # Integration defaults.
        self.camera_topic = "/csi_cam_0/image_raw"
        self.lidar_topic = "/scan"
        self.arm_topic = "/smart_city/arm"
        self.estop_topic = "/smart_city/estop"

        # Live mode refuses defaults until the team deliberately marks its own
        # measured file.  The example JSON leaves this False.
        self.calibrated = False
        self.calibration_id = ""

        self.validate()

    def update(self, values):
        """Apply known keys from a dict and reject accidental misspellings."""
        for key in values:
            if not hasattr(self, key):
                raise ValueError("Unknown Smart City setting: %s" % key)
        original = {key: getattr(self, key) for key in values}
        for key, value in values.items():
            setattr(self, key, value)
        try:
            self.validate()
        except (OverflowError, TypeError, ValueError):
            for key, value in original.items():
                setattr(self, key, value)
            raise
        return self

    def validate(self):
        """Validate general ranges used by both shadow and live modes."""
        string_fields = (
            "camera_topic", "lidar_topic", "arm_topic", "estop_topic",
            "calibration_id",
        )
        bool_fields = ("orange_keepout_enabled", "calibrated", "bench_only")
        hsv_fields = (
            "white_hsv_lower", "white_hsv_upper", "green_hsv_lower",
            "green_hsv_upper", "orange_hsv_lower_1", "orange_hsv_upper_1",
            "orange_hsv_lower_2", "orange_hsv_upper_2",
        )
        for name, value in self.__dict__.items():
            if name in string_fields:
                if not isinstance(value, str):
                    raise TypeError("%s must be a string" % name)
            elif name in bool_fields:
                if not isinstance(value, bool):
                    raise TypeError("%s must be boolean" % name)
            elif name in hsv_fields:
                _validate_hsv(value, name)
            else:
                _finite_number(value, name)
        for lower_name, upper_name in (
            ("white_hsv_lower", "white_hsv_upper"),
            ("green_hsv_lower", "green_hsv_upper"),
            ("orange_hsv_lower_1", "orange_hsv_upper_1"),
            ("orange_hsv_lower_2", "orange_hsv_upper_2"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if any(float(low) > float(high) for low, high in zip(lower, upper)):
                raise ValueError("%s must not exceed %s" % (
                    lower_name, upper_name
                ))

        for name in (
            "cruise_throttle", "approach_throttle", "turn_throttle",
            "reacquire_throttle", "straight_throttle",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError("%s must be in [0, 1]" % name)

        for name in ("turn_steering_left", "turn_steering_right"):
            value = float(getattr(self, name))
            if value < -1.0 or value > 1.0:
                raise ValueError("%s must be in [-1, 1]" % name)

        ratios = (
            "lane_roi_top", "lane_roi_bottom", "scan_near_ratio",
            "scan_far_ratio", "lane_min_width_ratio", "lane_max_width_ratio",
            "lane_default_width_ratio", "lane_min_confidence",
            "stop_roi_top", "stop_roi_bottom",
            "stop_max_component_area_ratio", "stop_min_span_ratio",
            "stop_y_backtrack_tolerance_ratio", "stop_approach_y_ratio",
            "stop_close_y_ratio", "green_roi_top",
            "green_roi_bottom", "green_danger_ratio",
            "green_bias_start_ratio", "turn_green_hard_stop_ratio",
            "max_lane_steering",
            "reacquire_center_error_ratio", "reacquire_heading_error_ratio",
            "ai_min_confidence",
        )
        for name in ratios:
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError("%s must be in [0, 1]" % name)

        ordered_pairs = (
            ("lane_roi_top", "lane_roi_bottom"),
            ("stop_roi_top", "stop_roi_bottom"),
            ("green_roi_top", "green_roi_bottom"),
            ("lane_min_width_ratio", "lane_max_width_ratio"),
        )
        for lower_name, upper_name in ordered_pairs:
            if float(getattr(self, lower_name)) >= float(getattr(self, upper_name)):
                raise ValueError("%s must be below %s" % (lower_name, upper_name))
        if float(self.scan_far_ratio) >= float(self.scan_near_ratio):
            raise ValueError("scan_far_ratio must be below scan_near_ratio")
        if not (
            float(self.stop_roi_top)
            <= float(self.stop_approach_y_ratio)
            < float(self.stop_close_y_ratio)
            <= float(self.stop_roi_bottom)
        ):
            raise ValueError(
                "stop y ratios must satisfy ROI top <= approach < close <= ROI bottom"
            )

        positive = (
            "loop_hz", "camera_timeout_seconds", "actuator_watchdog_seconds",
            "sensor_acquire_timeout_seconds",
            "arm_request_ttl_seconds",
            "lane_loss_stop_seconds", "lane_loss_estop_seconds",
            "turn_max_seconds", "nudge_max_seconds",
            "reacquire_timeout_seconds",
            "exit_lockout_max_seconds", "semantic_ttl_seconds",
            "red_light_timeout_seconds", "turn_min_seconds",
            "turn_nominal_seconds", "straight_cross_seconds",
            "steering_slew_per_second", "lidar_stop_distance_m",
            "lidar_timeout_seconds", "lidar_guard_half_angle_deg",
            "bench_max_runtime_seconds",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError("%s must be positive" % name)

        integer_fields = (
            "frame_width", "frame_height", "morph_kernel", "lane_min_pixels",
            "stop_min_components", "stop_min_component_area",
            "stop_cluster_y_px",
            "stop_confirm_frames", "initial_lane_stable_frames",
            "green_danger_confirm_frames", "turn_lane_confirm_frames",
            "nudge_marker_clear_frames", "reacquire_stable_frames",
            "exit_clear_frames", "ai_confirm_frames",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if int(value) != float(value) or int(value) < 1:
                raise ValueError("%s must be at least 1" % name)

        non_negative = (
            "stop_hold_seconds", "nudge_left_seconds", "nudge_right_seconds",
            "intersection_cooldown_seconds", "lane_kp", "lane_kd",
            "heading_gain", "green_avoidance_gain",
        )
        for name in non_negative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError("%s must be non-negative" % name)

        if not (
            float(self.turn_min_seconds)
            <= float(self.turn_nominal_seconds)
            <= float(self.turn_max_seconds)
        ):
            raise ValueError("turn timing must satisfy min <= nominal <= max")
        if float(self.nudge_max_seconds) < max(
            float(self.nudge_left_seconds), float(self.nudge_right_seconds)
        ):
            raise ValueError("nudge max must cover left/right nudge minimum")
        if float(self.lane_loss_stop_seconds) > float(self.lane_loss_estop_seconds):
            raise ValueError("lane loss stop must not exceed lane loss E-stop")
        if float(self.exit_lockout_max_seconds) < float(
            self.intersection_cooldown_seconds
        ):
            raise ValueError("exit lockout max must cover cooldown")
        if float(self.lidar_guard_half_angle_deg) > 90.0:
            raise ValueError("lidar guard half-angle must be <= 90 degrees")
        return self

    def validate_live(self):
        """Independent hard envelope for motor-enabled experiments."""
        self.validate()
        if self.bench_only:
            raise ValueError("bench-only config cannot be used for live competition")
        if self.calibrated is not True or not str(self.calibration_id).strip():
            raise ValueError(
                "live config needs calibrated=true and a calibration_id"
            )
        for name in (
            "cruise_throttle", "approach_throttle", "turn_throttle",
            "reacquire_throttle", "straight_throttle",
        ):
            if float(getattr(self, name)) > 0.30:
                raise ValueError("live throttle exceeds hard cap: %s" % name)
        if abs(float(self.turn_steering_left)) > 0.95:
            raise ValueError("turn_steering_left exceeds live hard cap")
        if abs(float(self.turn_steering_right)) > 0.95:
            raise ValueError("turn_steering_right exceeds live hard cap")
        if abs(float(self.turn_steering_left)) < 0.30:
            raise ValueError("turn_steering_left is too small for live mode")
        if abs(float(self.turn_steering_right)) < 0.30:
            raise ValueError("turn_steering_right is too small for live mode")
        if (
            float(self.turn_steering_left)
            * float(self.turn_steering_right)
            >= 0.0
        ):
            raise ValueError("live left/right turn steering must have opposite signs")
        if float(self.turn_max_seconds) > 2.50:
            raise ValueError("turn_max_seconds exceeds live hard cap")
        if float(self.straight_cross_seconds) > 2.00:
            raise ValueError("straight_cross_seconds exceeds live hard cap")
        if float(self.camera_timeout_seconds) > 0.50:
            raise ValueError("camera_timeout_seconds exceeds live hard cap")
        if float(self.actuator_watchdog_seconds) > 0.30:
            raise ValueError("actuator_watchdog_seconds exceeds live hard cap")
        if float(self.actuator_watchdog_seconds) < 0.05:
            raise ValueError("actuator_watchdog_seconds is unrealistically low")
        if float(self.arm_request_ttl_seconds) > 2.0:
            raise ValueError("live arm request TTL exceeds hard cap")
        if not 10.0 <= float(self.loop_hz) <= 40.0:
            raise ValueError("live loop_hz must be in [10, 40]")
        if not 0.40 <= float(self.stop_hold_seconds) <= 1.50:
            raise ValueError("live stop_hold_seconds must be in [0.40, 1.50]")
        for name in ("nudge_left_seconds", "nudge_right_seconds"):
            if float(getattr(self, name)) > 0.65:
                raise ValueError("live nudge exceeds hard cap: %s" % name)
        if float(self.nudge_max_seconds) > 2.0:
            raise ValueError("live nudge_max_seconds exceeds hard cap")
        if float(self.turn_min_seconds) < 0.40:
            raise ValueError("live turn_min_seconds must be at least 0.40")
        if float(self.reacquire_timeout_seconds) > 3.0:
            raise ValueError("live reacquire timeout exceeds hard cap")
        if float(self.exit_lockout_max_seconds) > 6.0:
            raise ValueError("live exit lockout exceeds hard cap")
        if float(self.lidar_timeout_seconds) > 0.50:
            raise ValueError("live LiDAR timeout exceeds hard cap")
        if float(self.lane_loss_estop_seconds) > 1.0:
            raise ValueError("live lane-loss timeout exceeds hard cap")
        if float(self.semantic_ttl_seconds) > 0.50:
            raise ValueError("live semantic TTL exceeds 0.50 second hard cap")
        if not 5.0 <= float(self.lidar_guard_half_angle_deg) <= 60.0:
            raise ValueError("live LiDAR guard half-angle must be in [5, 60]")
        if float(self.max_lane_steering) > 0.90:
            raise ValueError("live lane steering exceeds hard cap")
        if not 0.10 <= float(self.lane_min_confidence) <= 0.80:
            raise ValueError("live lane_min_confidence must be in [0.10, 0.80]")
        if float(self.reacquire_center_error_ratio) > 0.30:
            raise ValueError("live reacquire center tolerance exceeds 0.30")
        if float(self.reacquire_heading_error_ratio) > 0.25:
            raise ValueError("live reacquire heading tolerance exceeds 0.25")
        if float(self.intersection_cooldown_seconds) < 0.50:
            raise ValueError("live intersection cooldown must be at least 0.50")
        if float(self.steering_slew_per_second) > 6.0:
            raise ValueError("live steering slew exceeds hard cap")
        if not 160 <= int(self.frame_width) <= 1920:
            raise ValueError("live frame_width must be in [160, 1920]")
        if not 120 <= int(self.frame_height) <= 1080:
            raise ValueError("live frame_height must be in [120, 1080]")
        if not 1 <= int(self.morph_kernel) <= 15:
            raise ValueError("live morph_kernel must be in [1, 15]")
        if int(self.stop_min_components) < 4:
            raise ValueError("live stop-line detector needs at least 4 bars")
        if int(self.lane_min_pixels) < 4:
            raise ValueError("live lane_min_pixels must be at least 4")
        if float(self.stop_min_span_ratio) < 0.20:
            raise ValueError("live stop-line span ratio is too small")
        if not 0.15 <= float(self.lidar_stop_distance_m) <= 1.0:
            raise ValueError("live lidar stop distance must be in [0.15, 1.0]")
        if not 0.01 <= float(self.green_danger_ratio) <= 0.35:
            raise ValueError("live green danger ratio must be in [0.01, 0.35]")
        if not (
            float(self.green_danger_ratio)
            <= float(self.turn_green_hard_stop_ratio)
            <= 0.30
        ):
            raise ValueError(
                "live turn green hard-stop ratio must be between danger and 0.30"
            )
        if not 0.50 <= float(self.ai_min_confidence) <= 1.0:
            raise ValueError("live AI confidence must be in [0.50, 1.0]")
        minima = (
            ("stop_confirm_frames", 3),
            ("initial_lane_stable_frames", 3),
            ("ai_confirm_frames", 3),
            ("turn_lane_confirm_frames", 3),
            ("nudge_marker_clear_frames", 3),
            ("reacquire_stable_frames", 3),
            ("exit_clear_frames", 3),
            ("green_danger_confirm_frames", 2),
        )
        for name, minimum in minima:
            if int(getattr(self, name)) < minimum:
                raise ValueError("live %s must be >= %d" % (name, minimum))
        return self

    def validate_camera_bench(self):
        """Hard envelope for a supervised LEFT-then-RIGHT camera-only test.

        This mode intentionally does not claim that the course is calibrated.
        It permits no AI/LiDAR fallback and therefore uses stricter throttle,
        timing and total-runtime caps than normal live validation.
        """
        self.validate()
        if self.bench_only is not True:
            raise ValueError("camera bench config needs bench_only=true")
        if self.calibrated is not False:
            raise ValueError("camera bench config must not claim calibrated=true")
        if float(self.bench_max_runtime_seconds) > 30.0:
            raise ValueError("camera bench runtime exceeds 30 second hard cap")
        for name in (
            "cruise_throttle", "approach_throttle", "turn_throttle",
            "reacquire_throttle", "straight_throttle",
        ):
            if float(getattr(self, name)) > 0.12:
                raise ValueError("camera bench throttle exceeds 0.12: %s" % name)
        if float(self.max_lane_steering) > 0.80:
            raise ValueError("camera bench lane steering exceeds 0.80")
        if abs(float(self.turn_steering_left)) > 0.80:
            raise ValueError("camera bench left steering exceeds 0.80")
        if abs(float(self.turn_steering_right)) > 0.80:
            raise ValueError("camera bench right steering exceeds 0.80")
        if not float(self.turn_steering_left) <= -0.30:
            raise ValueError("camera bench LEFT steering must be <= -0.30")
        if not float(self.turn_steering_right) >= 0.30:
            raise ValueError("camera bench RIGHT steering must be >= 0.30")
        if not 10.0 <= float(self.loop_hz) <= 40.0:
            raise ValueError("camera bench loop_hz must be in [10, 40]")
        if not 0.40 <= float(self.stop_hold_seconds) <= 1.50:
            raise ValueError("camera bench stop hold must be in [0.40, 1.50]")
        for name in ("nudge_left_seconds", "nudge_right_seconds"):
            if float(getattr(self, name)) > 0.50:
                raise ValueError("camera bench nudge exceeds 0.50: %s" % name)
        if float(self.turn_max_seconds) > 1.80:
            raise ValueError("camera bench turn timeout exceeds 1.80 seconds")
        if float(self.nudge_max_seconds) > 1.50:
            raise ValueError("camera bench nudge timeout exceeds 1.50 seconds")
        if float(self.reacquire_timeout_seconds) > 2.50:
            raise ValueError("camera bench reacquire timeout exceeds 2.50 seconds")
        if float(self.exit_lockout_max_seconds) > 4.00:
            raise ValueError("camera bench exit lockout exceeds 4.00 seconds")
        if float(self.camera_timeout_seconds) > 0.30:
            raise ValueError("camera bench camera timeout exceeds 0.30 second")
        if not 0.05 <= float(self.actuator_watchdog_seconds) <= 0.20:
            raise ValueError("camera bench actuator watchdog must be [0.05, 0.20]")
        if float(self.lane_loss_estop_seconds) > 0.80:
            raise ValueError("camera bench lane-loss timeout exceeds 0.80 second")
        if not 0.10 <= float(self.lane_min_confidence) <= 0.80:
            raise ValueError("camera bench lane confidence must be in [0.10, 0.80]")
        if not 0.01 <= float(self.green_danger_ratio) <= 0.30:
            raise ValueError("camera bench green danger ratio is invalid")
        if not (
            float(self.green_danger_ratio)
            <= float(self.turn_green_hard_stop_ratio)
            <= 0.30
        ):
            raise ValueError("camera bench green hard-stop ratio is invalid")
        if int(self.initial_lane_stable_frames) < 5:
            raise ValueError("camera bench needs at least 5 stable lane frames")
        if int(self.stop_confirm_frames) < 3:
            raise ValueError("camera bench needs at least 3 stop-marker frames")
        for name, minimum in (
            ("nudge_marker_clear_frames", 3),
            ("reacquire_stable_frames", 3),
            ("exit_clear_frames", 3),
            ("green_danger_confirm_frames", 2),
        ):
            if int(getattr(self, name)) < minimum:
                raise ValueError(
                    "camera bench %s must be >= %d" % (name, minimum)
                )
        return self


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be numeric" % name)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("%s must be a finite number" % name)
    if not math.isfinite(number):
        raise ValueError("%s must be finite" % name)
    return number


def _validate_hsv(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError("%s must contain exactly three HSV numbers" % name)
    for index, item in enumerate(value):
        number = _finite_number(item, name)
        upper = 179.0 if index == 0 else 255.0
        if number < 0.0 or number > upper:
            raise ValueError("%s has a channel outside HSV range" % name)
