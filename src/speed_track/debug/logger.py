#!/usr/bin/env python3
"""
V3 Debug — Extended CSV Logger

Logs per-frame telemetry to CSV for post-analysis.
Format includes all key signals from perception, estimation, and control.
"""

import os
import csv
import time


class V3Logger:
    """Per-frame CSV logger for V3 pipeline telemetry."""

    FIELDNAMES = [
        'timestamp', 'fps',
        'left_conf', 'center_conf', 'right_conf',
        'lane_width_m',
        'lateral_error_m', 'heading_error_deg', 'curvature',
        'lookahead_m',
        'target_speed', 'actual_speed',
        'steer_raw', 'steer_filtered',
        'tracking_state',
        'reconstruction_method',
    ]

    def __init__(self, log_dir, prefix='v3'):
        """
        Args:
            log_dir: Directory to save log files.
            prefix: Filename prefix.
        """
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'{prefix}_{ts}.csv')

        self._file = open(self.log_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

        self._frame_count = 0
        self._fps_start = time.time()
        self._current_fps = 0.0

    def log(self, lane_state, trajectory_result, steer_raw, steer_filtered,
            throttle, actual_speed=0.0):
        """Log one frame of telemetry.

        Args:
            lane_state: LaneState from estimator.
            trajectory_result: TrajectoryResult from trajectory generator.
            steer_raw: Raw steering command.
            steer_filtered: Filtered steering command.
            throttle: Throttle command.
            actual_speed: Actual speed from encoder (m/s).
        """
        import math

        # Update FPS counter
        self._frame_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start = time.time()

        row = {
            'timestamp': f'{time.time():.3f}',
            'fps': f'{self._current_fps:.1f}',
            'left_conf': f'{lane_state.left_conf:.3f}',
            'center_conf': f'{lane_state.center_conf:.3f}',
            'right_conf': f'{lane_state.right_conf:.3f}',
            'lane_width_m': f'{lane_state.lane_width_m:.4f}',
            'lateral_error_m': f'{lane_state.lateral_error_m:.4f}',
            'heading_error_deg': f'{math.degrees(lane_state.heading_error):.2f}',
            'curvature': f'{lane_state.curvature:.5f}',
            'lookahead_m': f'{trajectory_result.lookahead_m:.3f}' if trajectory_result else '0.000',
            'target_speed': f'{throttle:.3f}',
            'actual_speed': f'{actual_speed:.3f}',
            'steer_raw': f'{steer_raw:.4f}',
            'steer_filtered': f'{steer_filtered:.4f}',
            'tracking_state': lane_state.tracking_state.name,
            'reconstruction_method': lane_state.reconstruction_method,
        }
        self._writer.writerow(row)

        # Flush periodically
        if self._frame_count % 20 == 0:
            self._file.flush()

    def get_fps(self):
        """Return current FPS estimate."""
        return self._current_fps

    def close(self):
        """Close the log file."""
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
