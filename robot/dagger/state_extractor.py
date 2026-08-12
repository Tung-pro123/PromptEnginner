#!/usr/bin/env python3
"""
DAgger — State Extractor
========================
Trích xuất vector trạng thái S_t từ:
  - LaneState (lateral_error, heading_error, line_visible)
  - LaserScan  (5 LiDAR zones + obstacle_detected flag)

Vector S_t shape: (9,)
  [0]  e_y              : lateral error (m), chuẩn hoá về [-1, 1] theo E_LAT_MAX
  [1]  theta_e          : heading error (rad), chuẩn hoá về [-1, 1] theo THETA_MAX
  [2]  line_visible     : 1.0 nếu tracking, 0.0 nếu mất vạch
  [3]  d_left           : khoảng cách zone trái (0°–36°), chuẩn hoá [0,1]
  [4]  d_front_left     : khoảng cách zone trái-trước (36°–72°)
  [5]  d_front          : khoảng cách zone trước (72°–108°), quan trọng nhất
  [6]  d_front_right    : khoảng cách zone phải-trước (108°–144°)
  [7]  d_right          : khoảng cách zone phải (144°–180°)
  [8]  obstacle_detected: 1.0 nếu có vật cản trước mặt (d_front < OBSTACLE_THRESH), else 0.0

Quy ước LiDAR angles: phụ thuộc vào LIDAR_OFFSET_DEG của xe
  - 0° = phía trước xe, tăng dần theo chiều kim đồng hồ
  - LIDAR_OFFSET_DEG = 180° (như trong main_speed_track_v3.py)
"""

import math
import numpy as np

# =====================================================================
# CONSTANTS (chỉnh tại đây nếu cần)
# =====================================================================
STATE_DIM      = 9        # tổng số chiều của S_t

E_LAT_MAX      = 0.30     # (m) lateral error tối đa để normalize → [-1, 1]
THETA_MAX      = 0.60     # (rad) ≈ 34°, heading error tối đa

LIDAR_OFFSET_DEG = 180.0  # khớp với main_speed_track_v3.py
LIDAR_MAX_RANGE  = 1.5    # (m) chuẩn hoá d: 1.5m → 1.0
OBSTACLE_THRESH  = 0.50   # (m) d_front < ngưỡng này → obstacle_detected = 1

# 5 zones, mỗi zone 36° — phủ kín 180° phía trước
# Tính theo góc thực của LiDAR (sau khi đã áp LIDAR_OFFSET_DEG)
# Zone angles (deg): trái → phải
ZONE_ANGLES = [
    (0,   36),   # zone 0: left
    (36,  72),   # zone 1: front-left
    (72,  108),  # zone 2: front  ← quan trọng nhất
    (108, 144),  # zone 3: front-right
    (144, 180),  # zone 4: right
]
N_ZONES = len(ZONE_ANGLES)


def extract_lidar_zones(scan_msg):
    """
    Trích xuất N_ZONES khoảng cách min từ LaserScan.

    Args:
        scan_msg: sensor_msgs/LaserScan hoặc None

    Returns:
        zones (np.ndarray shape (N_ZONES,)) — normalized [0, 1]
        obstacle_detected (float)           — 1.0 hoặc 0.0
    """
    zones = np.ones(N_ZONES, dtype=np.float32)  # default = 1.0 (xa, an toàn)

    if scan_msg is None:
        return zones, 0.0

    n = len(scan_msg.ranges)
    angle_min_deg = math.degrees(scan_msg.angle_min)
    angle_inc_deg = math.degrees(scan_msg.angle_increment)

    zone_mins = [LIDAR_MAX_RANGE] * N_ZONES

    for i, raw_d in enumerate(scan_msg.ranges):
        # Bỏ qua giá trị không hợp lệ
        if not (scan_msg.range_min < raw_d < scan_msg.range_max):
            continue
        d = min(raw_d, LIDAR_MAX_RANGE)

        # Góc thực tính toán (deg)
        raw_deg = angle_min_deg + i * angle_inc_deg
        deg = (raw_deg + LIDAR_OFFSET_DEG) % 360  # chuyển về frame của xe

        # Chỉ quan tâm 0°–180° (nửa trước xe)
        if not (0 <= deg <= 180):
            continue

        for z, (lo, hi) in enumerate(ZONE_ANGLES):
            if lo <= deg < hi:
                if d < zone_mins[z]:
                    zone_mins[z] = d
                break

    # Normalize: d / LIDAR_MAX_RANGE → [0, 1]
    zones = np.array([d / LIDAR_MAX_RANGE for d in zone_mins], dtype=np.float32)

    # obstacle_detected: zone trước (zone 2) chạm ngưỡng
    front_dist = zone_mins[2]  # raw meters, zone front
    obstacle_detected = 1.0 if front_dist < OBSTACLE_THRESH else 0.0

    return zones, obstacle_detected


def extract_state(lane_state, scan_msg):
    """
    Build vector S_t từ LaneState + LaserScan.

    Args:
        lane_state: implements.estimation.lane_state.LaneState (hoặc None)
        scan_msg  : sensor_msgs/LaserScan (hoặc None)

    Returns:
        state_vec (np.ndarray shape (STATE_DIM,), dtype=float32)
        info (dict) — debug info
    """
    # ---- Vision features ----
    if lane_state is not None:
        # e_y: lateral error, normalize và clamp [-1, 1]
        e_y = float(lane_state.lateral_error_m) / E_LAT_MAX
        e_y = max(-1.0, min(1.0, e_y))

        # theta_e: heading error, normalize
        theta_e = float(lane_state.heading_error) / THETA_MAX
        theta_e = max(-1.0, min(1.0, theta_e))

        # line_visible: xét tracking state
        from robot.estimation.lane_state import TrackingState
        ts = lane_state.tracking_state
        line_visible = 1.0 if ts in (
            TrackingState.TRACKING, TrackingState.UNCERTAIN
        ) else 0.0
    else:
        e_y         = 0.0
        theta_e     = 0.0
        line_visible = 0.0

    # ---- LiDAR features ----
    lidar_zones, obstacle_detected = extract_lidar_zones(scan_msg)

    # ---- Assemble S_t ----
    state_vec = np.array(
        [e_y, theta_e, line_visible] + lidar_zones.tolist() + [obstacle_detected],
        dtype=np.float32
    )

    info = {
        'e_y_raw'          : lane_state.lateral_error_m if lane_state else 0.0,
        'theta_e_raw'      : lane_state.heading_error   if lane_state else 0.0,
        'line_visible'     : line_visible,
        'lidar_zones_m'    : lidar_zones * LIDAR_MAX_RANGE,
        'obstacle_detected': obstacle_detected,
    }

    return state_vec, info
