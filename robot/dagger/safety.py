#!/usr/bin/env python3
"""
DAgger — Safety Layer
=====================
Lớp an toàn chủ động dựa trên LiDAR zones.

Hoạt động độc lập với AI — luôn kiểm tra trước khi gửi lệnh xuống xe.

Ngưỡng:
  d_critical = 0.25 m → E-STOP (throttle = 0, áp phanh khẩn cấp)
  d_warn     = 0.50 m → giảm ga theo tỉ lệ (throttle * scale_factor)

Chỉ xét zone FRONT (zone 2) để E-STOP.
Xét tổng hợp các zone phía trước (1, 2, 3) để giảm ga.
"""

import numpy as np

# =====================================================================
# THRESHOLDS (chỉnh tại đây)
# =====================================================================
D_CRITICAL_M   = 0.25   # (m) — dưới mức này: E-STOP ngay
D_WARN_M       = 0.50   # (m) — dưới mức này: bắt đầu giảm ga
MIN_THROTTLE   = 0.10   # throttle tối thiểu khi đang cảnh báo (không về 0 hoàn toàn)

# Index của zone trong lidar_zones vector (khớp với state_extractor.py)
ZONE_FRONT       = 2    # d_front
ZONE_FRONT_LEFT  = 1    # d_front_left
ZONE_FRONT_RIGHT = 3    # d_front_right

# Scale: giảm throttle theo khoảng cách (linear interpolation)
# d = D_WARN   → throttle * 1.0  (không giảm)
# d = D_CRIT   → throttle * 0.0  (phanh hoàn toàn)


class SafetyLayer:
    """
    Kiểm tra an toàn và điều chỉnh (steer, throttle) trước khi gửi xuống xe.

    Usage:
        safety = SafetyLayer()
        safe_steer, safe_throttle, is_estop = safety.check(steer, throttle, lidar_zones_normalized)
    """

    def __init__(self,
                 d_critical_m=D_CRITICAL_M,
                 d_warn_m=D_WARN_M):
        self.d_critical = d_critical_m
        self.d_warn     = d_warn_m
        self._estop_active = False

    def check(self, steer: float, throttle: float, lidar_zones_normalized: np.ndarray):
        """
        Áp dụng safety filter lên lệnh điều khiển.

        Args:
            steer                   : steer ∈ [-1, 1]
            throttle                : throttle ∈ [0, 1]
            lidar_zones_normalized  : np.ndarray (5,) ∈ [0,1],
                                      đã normalize theo LIDAR_MAX_RANGE
                                      (từ state_extractor.extract_lidar_zones)

        Returns:
            safe_steer    (float)
            safe_throttle (float)
            is_estop      (bool)  — True nếu phanh khẩn cấp
        """
        from robot.dagger.state_extractor import LIDAR_MAX_RANGE

        # Chuyển normalized về metres để so sánh ngưỡng vật lý
        zones_m = lidar_zones_normalized * LIDAR_MAX_RANGE

        front_dist       = zones_m[ZONE_FRONT]
        front_left_dist  = zones_m[ZONE_FRONT_LEFT]
        front_right_dist = zones_m[ZONE_FRONT_RIGHT]

        # --- E-STOP: vật cản chính diện cực gần ---
        if front_dist < self.d_critical:
            self._estop_active = True
            return 0.0, 0.0, True

        # --- Cảnh báo: giảm ga khi tiếp cận ---
        min_front = min(front_dist, front_left_dist, front_right_dist)

        if min_front < self.d_warn:
            # Tỉ lệ: 1.0 tại d_warn, 0.0 tại d_critical
            ratio = (min_front - self.d_critical) / (self.d_warn - self.d_critical)
            ratio = max(0.0, min(1.0, ratio))

            # Scale throttle xuống, giữ tối thiểu MIN_THROTTLE nếu còn di chuyển
            safe_throttle = max(MIN_THROTTLE, throttle * ratio)
        else:
            safe_throttle = throttle

        self._estop_active = False
        return steer, safe_throttle, False

    @property
    def estop_active(self):
        return self._estop_active

    def reset(self):
        """Reset trạng thái sau khi đã xử lý E-STOP."""
        self._estop_active = False
