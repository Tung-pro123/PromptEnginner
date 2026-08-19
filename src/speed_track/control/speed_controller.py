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
        self._cycle_progress = 0.0
        self._prev_time_cycle = None

        # V3.1: Curvature history for stability detection
        self._curvature_history = []
        self._history_size = getattr(config, 'curvature_history_size', 10)
        self._stability_bonus = getattr(config, 'curvature_stability_bonus', 1.0)
        self._stability_thresh = getattr(config, 'curvature_stability_thresh', 0.1)
        
        # Smooth Throttle Control
        self.current_throttle = config.min_speed

    def compute(self, curvature, confidence, tracking_state, actual_speed=None, horizon_state="STRAIGHT"):
        """Compute target throttle.

        Args:
            curvature: Road curvature (1/m) from trajectory module.
            confidence: Lane confidence [0,1] from state estimator.
            tracking_state: TrackingState enum value.
            actual_speed: Actual speed from encoder (m/s). None if no encoder.
            horizon_state: [V3.1] "STRAIGHT", "CURVE_LEFT", or "CURVE_RIGHT" from Horizon Scanner.

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

        # V3.1: Use max of recent curvatures for safety (defensive driving)
        # but allow a stability bonus when curvature is consistent (oval track)
        stability_bonus = 1.0
        if len(self._curvature_history) >= 3:
            effective_curvature = max(self._curvature_history)
            std_recent = float(np.std(self._curvature_history))

            # Curvature is stable (low std) → car is on a consistent curve → safe to go faster
            if std_recent < self._stability_thresh and self._stability_bonus > 1.0:
                stability_bonus = self._stability_bonus
        else:
            effective_curvature = abs(curvature)

        v_curve = math.sqrt(cfg.a_lat_max / (effective_curvature + epsilon))
        v_target = min(v_curve * stability_bonus, cfg.cruise_speed)

        # V3.1: Explicit Horizon Control (Phóng nhanh / Đi chậm)
        if horizon_state in ["CURVE_LEFT", "CURVE_RIGHT"]:
            # Nhìn thấy cua gắt phía xa -> TRỞ VỀ BÁM ĐƯỜNG BÌNH THƯỜNG (không ép đi chậm)
            # Hệ thống sẽ dùng vận tốc bám đường (v_curve) vốn rất ổn định.
            pass
        elif horizon_state == "STRAIGHT" and effective_curvature < 0.3:
            # Nhìn thấy đường thẳng tắp VÀ gầm xe cũng đã thoát cua -> Đặt mục tiêu tốc độ tối đa!
            v_target = cfg.max_speed

        # Step 2: Confidence scaling
        if confidence < cfg.speed_confidence_thresh:
            # Linear reduction: at confidence=0 → v_target * 0
            scale = max(0.3, confidence / cfg.speed_confidence_thresh)
            v_target *= scale

        # Step 3: State machine speed limits
        if tracking_state == TrackingState.SEARCH:
            v_target = cfg.min_speed
        elif tracking_state == TrackingState.UNCERTAIN:
            v_target = min(v_target, cfg.cruise_speed * 0.7)
        elif tracking_state == TrackingState.PREDICTING:
            v_target = cfg.min_speed
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
        if getattr(cfg, 'use_cyclic_throttle', False):
            # Vận hành theo chu trình (Cyclic Throttle)
            import time
            current_time = time.time()
            if self._prev_time_cycle is None:
                dt = 0.05
            else:
                dt = current_time - self._prev_time_cycle
            self._prev_time_cycle = current_time
            
            # Kiểm tra xem đang đi thẳng hay vào cua
            curve_thresh = getattr(cfg, 'curve_threshold', 0.15)
            is_curve = (effective_curvature > curve_thresh) or (horizon_state in ["CURVE_LEFT", "CURVE_RIGHT"])
            
            if is_curve:
                cycle_duration = getattr(cfg, 'cycle_duration_curve', 1.0)
            else:
                cycle_duration = getattr(cfg, 'cycle_duration_straight', 2.0)
            
            # Cập nhật tiến trình (phase) liên tục để không bị nhảy cóc khi duration thay đổi
            self._cycle_progress += dt / cycle_duration
            if self._cycle_progress >= 1.0:
                self._cycle_progress -= 1.0
            elif self._cycle_progress < 0.0:
                self._cycle_progress = 0.0
            
            # Chu trình: Giảm dần từ max_speed xuống min_speed
            cyclic_throttle = cfg.max_speed - self._cycle_progress * (cfg.max_speed - cfg.min_speed)
            
            # CẢI TIẾN: Chỉ dùng Cyclic Throttle làm "mức trần" (trên đường thẳng).
            # Vẫn tính v_curve để nếu tới cua gắt, xe bắt buộc phải hãm lại theo đường cong.
            # (Đảm bảo an toàn vật lý khi ôm cua)
            safe_curve_throttle = v_target * cfg.speed_to_throttle_factor
            throttle = min(cyclic_throttle, safe_curve_throttle)
            
            # Vẫn giữ bảo vệ giảm tốc khi mất vạch (Confidence) và State Machine
            if confidence < cfg.speed_confidence_thresh:
                throttle *= max(0.3, confidence / cfg.speed_confidence_thresh)
                
            if tracking_state in [TrackingState.SEARCH, TrackingState.PREDICTING, TrackingState.RECOVERY]:
                throttle = cfg.min_speed
            elif tracking_state == TrackingState.E_STOP:
                throttle = 0.0
                
        else:
            if cfg.use_encoder and actual_speed is not None:
                # Closed-loop PID
                throttle = self._pid_compute(v_target, actual_speed)
            else:
                # Open-loop mapping
                throttle = v_target * cfg.speed_to_throttle_factor

        # Final clamp
        target_throttle = max(0.0, min(cfg.max_speed, throttle))

        # V3.1: Mượt mà hóa chân ga (Smooth Throttle)
        # BỎ QUA LÀM MƯỢT TRONG TRƯỜNG HỢP KHẨN CẤP!
        if tracking_state in [TrackingState.E_STOP, TrackingState.SEARCH]:
            self.current_throttle = target_throttle
            return self.current_throttle

        # Tăng ga từ từ (alpha nhỏ), nhưng hạ ga thật nhanh khi gặp cua (alpha lớn)
        if target_throttle > self.current_throttle:
            alpha = 0.05 # Tăng ga mượt mà
        else:
            alpha = 0.3  # Hạ ga nhanh để an toàn
            
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
