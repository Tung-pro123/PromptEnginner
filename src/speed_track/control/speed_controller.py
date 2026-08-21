#!/usr/bin/env python3
"""
V3 / V3.1 Control — Curvature-Based Speed Controller

Computes target speed based on:
1. Road curvature:  v_curve = sqrt(a_lat_max / (|kappa| + eps))
2. Lane confidence: reduce speed when perception is uncertain
3. Tracking state: reduce/stop based on state machine

V3.1 additions (backward compatible):
4. Curvature history: tracks N recent curvature values
5. Stability bonus: if curvature is stable (low std), allows higher speed
   This is especially effective on oval tracks where curvature is nearly constant.

Supports optional encoder-based PID for closed-loop speed control.
Falls back to open-loop throttle mapping when no encoder is available.

V2 used a fixed px_to_m scalar and got incorrect v_max values.
V3 uses proper metric curvature from the trajectory module.
"""

import math
import time
import numpy as np


class SpeedController:
    """Curvature-aware speed controller with optional PID."""

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.cfg = config

        # PID state (for encoder feedback mode)
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

        # V3.1: Curvature history for stability detection
        self._curvature_history = []
        self._history_size = getattr(config, 'curvature_history_size', 10)
        self._stability_bonus = getattr(config, 'curvature_stability_bonus', 1.0)
        self._stability_thresh = getattr(config, 'curvature_stability_thresh', 0.1)
        
        # Smooth Throttle Control
        self.current_throttle = config.min_speed
        
        # Straight Boost State (Debounce & Timeout)
        self._straight_frames = 0
        self._is_boosting = False
        self._boost_start_time = 0.0

    def compute(self, curvature, confidence, tracking_state, actual_speed=None, horizon_state="STRAIGHT", max_upcoming_curvature=None, heading_error=None):
        """Compute target throttle with Inflection Point Protection.

        Args:
            curvature: Road curvature (1/m) from trajectory module.
            confidence: Lane confidence [0,1] from state estimator.
            tracking_state: TrackingState enum value.
            actual_speed: Actual speed from encoder (m/s). None if no encoder.
            horizon_state: [V3.1] "STRAIGHT", "CURVE_LEFT", or "CURVE_RIGHT".
            max_upcoming_curvature: [V3.3] Peak curvature ahead on lookahead trajectory.
            heading_error: Heading error (radians) at vehicle position.

        Returns:
            Throttle command (0.0 to max_speed).
        """
        from src.speed_track.estimation.lane_state import TrackingState

        cfg = self.cfg

        # V3.1: Track curvature history
        self._curvature_history.append(abs(curvature))
        if len(self._curvature_history) > self._history_size:
            self._curvature_history.pop(0)

        # Step 1: Curvature-based target speed
        epsilon = 1e-4

        stability_bonus = 1.0
        if len(self._curvature_history) >= 3:
            effective_curvature = max(self._curvature_history)
            std_recent = float(np.std(self._curvature_history))

            if std_recent < self._stability_thresh and self._stability_bonus > 1.0:
                stability_bonus = self._stability_bonus
        else:
            effective_curvature = abs(curvature)

        v_curve = math.sqrt(cfg.a_lat_max / (effective_curvature + epsilon))
        v_target = min(v_curve * stability_bonus, cfg.max_speed)

        # V3.3: PREDICTIVE CORNER BRAKING & STRAIGHTAWAY QUALIFICATION
        corner_thresh = getattr(cfg, 'corner_brake_curvature_thresh', 0.65)
        corner_safe_v = getattr(cfg, 'corner_safe_speed', 0.34)

        # Kết hợp độ cong sát xe và độ cong dự đoán phía trước (trong tầm 1.2m)
        lookahead_curvature = max(effective_curvature, abs(max_upcoming_curvature)) if max_upcoming_curvature is not None else effective_curvature

        # STRAIGHTAWAY QUALIFICATION FILTER (Chỉ mở 90% ga trên đoạn thẳng dài 05 -> 04)
        if lookahead_curvature < 0.25:
            self._straight_frames = getattr(self, '_straight_frames', 0) + 1
        else:
            self._straight_frames = 0

        # Cần ít nhất 8 frames liên tiếp đi thẳng (~0.6s) thì mới được công nhận là đoạn thẳng dài
        is_long_straight = self._straight_frames >= 8

        # V3.3: S-ZONE CHICANE LATCH (Khóa ga an toàn trong toàn bộ phân đoạn chữ S [02 -> 03 -> 02])
        # Khi phát hiện độ cong >= 0.55, kích hoạt khóa ga an toàn trong 28 frames (~2.2s)
        if lookahead_curvature >= 0.55:
            self._s_zone_hold_frames = 28
        else:
            self._s_zone_hold_frames = max(0, getattr(self, '_s_zone_hold_frames', 0) - 1)

        in_s_zone = self._s_zone_hold_frames > 0

        if in_s_zone or lookahead_curvature >= corner_thresh:
            # 1. Toàn bộ khúc chữ S [02 -> 03 -> 02] -> GHIM CỨNG GA AN TOÀN (corner_safe_speed = 0.36)
            # TUYỆT ĐỐI KHÔNG BỨT TỐC GIỮA CHỮ S ĐỂ TRÁNH MẤT LINE!
            v_target = corner_safe_v
        elif is_long_straight:
            # 2. Đoạn thẳng dài thực sự (05 -> 04) -> Phóng ga tối đa 100%!
            v_target = cfg.max_speed
        else:
            # 3. Vòng cung lớn [01] -> cruise_speed (0.70)
            v_target = min(v_target, cfg.cruise_speed)

        # Step 2: Confidence scaling
        if confidence < cfg.speed_confidence_thresh:
            # Linear reduction: at confidence=0 → v_target * 0
            scale = max(0.4, confidence / cfg.speed_confidence_thresh)
            v_target *= scale

        # Step 3: State machine speed limits
        if tracking_state == TrackingState.SEARCH:
            v_target = cfg.min_speed
        elif tracking_state == TrackingState.UNCERTAIN:
            v_target = min(v_target, cfg.cruise_speed * 0.75)
        elif tracking_state == TrackingState.PREDICTING:
            v_target = max(cfg.min_speed, corner_safe_v * 0.85)
        elif tracking_state == TrackingState.RECOVERY:
            v_target = cfg.min_speed
        elif tracking_state == TrackingState.E_STOP:
            v_target = 0.0

        # Clamp to valid range
        if v_target > 0:
            v_target = max(cfg.min_speed, min(cfg.max_speed, v_target))
        else:
            v_target = 0.0

        # Step 4: Convert to throttle
        if cfg.use_encoder and actual_speed is not None:
            # Closed-loop PID
            throttle = self._pid_compute(v_target, actual_speed)
        else:
            # Open-loop mapping
            throttle = v_target * cfg.speed_to_throttle_factor

        # Final clamp
        target_throttle = max(0.0, min(cfg.max_speed, throttle))

        # V3.1 / V3.3: Mượt mà hóa chân ga (Smooth Throttle)
        if tracking_state in [TrackingState.E_STOP, TrackingState.SEARCH]:
            self.current_throttle = target_throttle
            return self.current_throttle

        # Tăng ga dứt khoát (alpha = 0.22) để thoát cua bứt tốc, hạ ga khẩn cấp (alpha = 0.50) khi vào cua
        if target_throttle > self.current_throttle:
            alpha = 0.22 # Tăng ga dứt khoát thoát cua bứt tốc
        else:
            alpha = 0.50 if lookahead_curvature >= corner_thresh else 0.30  # Phanh gấp khi vào cua
            
        self.current_throttle = (1.0 - alpha) * self.current_throttle + alpha * target_throttle

        return self.current_throttle

    def _pid_compute(self, target_speed, actual_speed):
        """PID control for speed (encoder feedback).

        Args:
            target_speed: Desired speed (m/s).
            actual_speed: Measured speed from encoder (m/s).

        Returns:
            Throttle command.
        """
        cfg = self.cfg
        now = time.time()

        error = target_speed - actual_speed

        if self._prev_time is None:
            dt = 0.05
        else:
            dt = max(0.01, now - self._prev_time)

        # P
        p = cfg.speed_pid_kp * error

        # I (with anti-windup)
        self._integral += error * dt
        self._integral = max(-1.0, min(1.0, self._integral))
        i = cfg.speed_pid_ki * self._integral

        # D
        d = cfg.speed_pid_kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time = now

        return p + i + d

    def reset(self):
        """Reset PID state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._curvature_history = []
        self.current_throttle = self.cfg.min_speed
