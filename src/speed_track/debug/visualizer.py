#!/usr/bin/env python3
"""
V3 Debug — Full Visualization

Renders a comprehensive debug dashboard showing:
- Original frame with ROI and mask overlay
- BEV with detected L/C/R lines, centerline, sliding windows, lookahead
- Text overlays: state, FPS, confidences, errors, speeds, steering

The dashboard is recorded to an AVI file for post-analysis.
"""

import cv2
import math
import numpy as np
from typing import Optional


class DebugVisualizer:
    """Renders debug visualization for the V3 pipeline."""

    # Colors (BGR)
    COLOR_LEFT = (255, 100, 100)     # Blue-ish for left boundary
    COLOR_CENTER = (0, 255, 255)     # Yellow for center dashed line
    COLOR_RIGHT = (100, 100, 255)    # Red-ish for right boundary
    COLOR_CENTERLINE = (0, 255, 0)   # Green for reconstructed center
    COLOR_LOOKAHEAD = (255, 0, 255)  # Magenta for lookahead point
    COLOR_ROI = (0, 255, 255)        # Yellow for ROI rectangle
    COLOR_TEXT = (255, 255, 255)     # White text
    COLOR_WARN = (0, 165, 255)       # Orange warning text
    COLOR_ERROR = (0, 0, 255)        # Red error text

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.cfg = config
        self.W = config.image_width
        self.H = config.image_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.45
        self.font_thickness = 1

    def render(self, frame, bev_mask, detection_result, lane_state,
               trajectory_result, steer_raw, steer_filtered, throttle,
               fps, actual_speed=0.0):
        """Render complete debug dashboard.

        Args:
            frame: Original BGR frame (640x480).
            bev_mask: Binary mask in BEV space.
            detection_result: LaneDetectionResult from lane detector.
            lane_state: LaneState from state estimator.
            trajectory_result: TrajectoryResult from trajectory generator.
            steer_raw: Raw steering command.
            steer_filtered: Filtered steering command.
            throttle: Current throttle command.
            fps: Current FPS.
            actual_speed: Actual speed from encoder (m/s).

        Returns:
            Combined dashboard image (1280x480) or None if inputs are invalid.
        """
        if frame is None:
            return None

        # Ensure correct dimensions
        if frame.shape[:2] != (self.H, self.W):
            frame = cv2.resize(frame, (self.W, self.H))

        # --- Left panel: Original frame with overlays ---
        left_panel = frame.copy()
        self._draw_roi(left_panel)
        self._draw_state_text(left_panel, lane_state, steer_filtered, throttle, fps)

        # --- Right panel: BEV with lane lines ---
        if bev_mask is not None:
            right_panel = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        else:
            right_panel = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        self._draw_detected_lines(right_panel, detection_result)
        self._draw_centerline(right_panel, lane_state)
        self._draw_trajectory(right_panel, trajectory_result)
        self._draw_metrics(right_panel, lane_state, trajectory_result,
                           steer_raw, steer_filtered, throttle, actual_speed)

        # --- Combine ---
        if right_panel.shape[:2] != (self.H, self.W):
            right_panel = cv2.resize(right_panel, (self.W, self.H))

        dashboard = np.hstack((left_panel, right_panel))
        return dashboard

    def _draw_roi(self, frame):
        """Draw ROI rectangle on original frame."""
        y_start = int(self.cfg.roi_y_start * self.H)
        cv2.rectangle(frame, (0, y_start), (self.W - 1, self.H - 1),
                       self.COLOR_ROI, 1)

        # Draw BEV source points
        pts = np.int32(self.cfg.bev_src_pts)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 1)

    def _draw_state_text(self, frame, lane_state, steer, throttle, fps):
        """Draw tracking state and key info on original frame."""
        state_name = lane_state.tracking_state.name if lane_state else 'INIT'

        # State-dependent color
        from src.speed_track.estimation.lane_state import TrackingState
        state_colors = {
            TrackingState.SEARCH: self.COLOR_WARN,
            TrackingState.TRACKING: (0, 255, 0),
            TrackingState.UNCERTAIN: self.COLOR_WARN,
            TrackingState.PREDICTING: self.COLOR_ERROR,
            TrackingState.RECOVERY: self.COLOR_ERROR,
            TrackingState.E_STOP: (0, 0, 200),
        }
        state_color = state_colors.get(lane_state.tracking_state, self.COLOR_TEXT) if lane_state else self.COLOR_TEXT

        y = 25
        cv2.putText(frame, f"State: {state_name}", (10, y),
                     self.font, 0.7, state_color, 2)
        y += 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                     self.font, 0.5, self.COLOR_TEXT, 1)
        y += 22
        cv2.putText(frame, f"Steer: {steer:.3f}", (10, y),
                     self.font, 0.5, self.COLOR_TEXT, 1)
        y += 22
        cv2.putText(frame, f"Throttle: {throttle:.3f}", (10, y),
                     self.font, 0.5, self.COLOR_TEXT, 1)

    def _draw_detected_lines(self, bev_viz, detection_result):
        """Draw detected L/C/R lines on BEV visualization."""
        if detection_result is None:
            return

        plot_y = np.linspace(0, self.H - 1, self.H)

        for line_det, color, label in [
            (detection_result.left, self.COLOR_LEFT, 'L'),
            (detection_result.center, self.COLOR_CENTER, 'C'),
            (detection_result.right, self.COLOR_RIGHT, 'R'),
        ]:
            if line_det.detected and line_det.poly is not None:
                plot_x = np.polyval(line_det.poly, plot_y)

                # Clip to image bounds
                valid = (plot_x >= 0) & (plot_x < self.W)
                if np.sum(valid) > 1:
                    pts = np.array(list(zip(
                        plot_x[valid].astype(int),
                        plot_y[valid].astype(int)
                    )), dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(bev_viz, [pts], False, color, 2)

                # Confidence label at top of line
                first_valid = np.argmax(valid)
                if first_valid < len(plot_x):
                    label_x = int(np.clip(plot_x[first_valid], 5, self.W - 40))
                    label_y = int(plot_y[first_valid]) + 15
                    cv2.putText(bev_viz,
                                f"{label}:{line_det.confidence:.2f}",
                                (label_x, label_y),
                                self.font, 0.4, color, 1)

    def _draw_centerline(self, bev_viz, lane_state):
        """Draw the estimated centerline on BEV."""
        if lane_state is None or lane_state.centerline_poly is None:
            return

        plot_y = np.linspace(0, self.H - 1, self.H)
        plot_x = np.polyval(lane_state.centerline_poly, plot_y)

        valid = (plot_x >= 0) & (plot_x < self.W)
        if np.sum(valid) > 1:
            pts = np.array(list(zip(
                plot_x[valid].astype(int),
                plot_y[valid].astype(int)
            )), dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(bev_viz, [pts], False, self.COLOR_CENTERLINE, 2)

    def _draw_trajectory(self, bev_viz, trajectory_result):
        """Draw trajectory points and lookahead target."""
        if trajectory_result is None:
            return

        # Draw trajectory points
        for pt in trajectory_result.points:
            x, y = int(pt.x_px), int(pt.y_px)
            if 0 <= x < self.W and 0 <= y < self.H:
                cv2.circle(bev_viz, (x, y), 3, self.COLOR_CENTERLINE, -1)

        # Draw target point (larger, different color)
        if trajectory_result.target is not None:
            tx = int(trajectory_result.target.x_px)
            ty = int(trajectory_result.target.y_px)
            if 0 <= tx < self.W and 0 <= ty < self.H:
                cv2.circle(bev_viz, (tx, ty), 8, self.COLOR_LOOKAHEAD, 2)
                cv2.circle(bev_viz, (tx, ty), 3, self.COLOR_LOOKAHEAD, -1)

                # Line from vehicle to target
                vx = self.W // 2
                vy = self.H
                cv2.line(bev_viz, (vx, vy), (tx, ty), self.COLOR_LOOKAHEAD, 1)

    def _draw_metrics(self, bev_viz, lane_state, trajectory_result,
                      steer_raw, steer_filtered, throttle, actual_speed):
        """Draw metric text on BEV panel."""
        if lane_state is None:
            return

        x_col = 10
        y = self.H - 160

        lines = [
            f"L:{lane_state.left_conf:.2f} C:{lane_state.center_conf:.2f} R:{lane_state.right_conf:.2f}",
            f"e_lat: {lane_state.lateral_error_m*100:.1f}cm",
            f"e_psi: {math.degrees(lane_state.heading_error):.1f}deg",
            f"kappa: {lane_state.curvature:.3f} 1/m",
            f"W: {lane_state.lane_width_m*100:.1f}cm",
            f"Ld: {trajectory_result.lookahead_m:.2f}m" if trajectory_result else "Ld: --",
            f"steer: {steer_raw:.3f} -> {steer_filtered:.3f}",
            f"method: {lane_state.reconstruction_method}",
        ]

        for line in lines:
            cv2.putText(bev_viz, line, (x_col, y),
                         self.font, self.font_scale, self.COLOR_TEXT,
                         self.font_thickness)
            y += 18
