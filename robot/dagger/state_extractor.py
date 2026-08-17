#!/usr/bin/env python3
"""
DAgger — State Extractor (V2 - 15 Dimensional Feature Space)
============================================================
Trích xuất vector trạng thái S_t đa chiều giàu ngữ cảnh từ:
  - LaneState (lateral_error, lateral_velocity, heading_error, curvature, line_visible)
  - LaserScan (5 LiDAR zones, side clearance diff, min front corridor, obstacle flag)
  - Action History (prev_steer, prev_throttle)

Vector S_t shape: (15,)
  [0]  e_y               : lateral error (m), chuẩn hoá [-1, 1] theo E_LAT_MAX
  [1]  e_y_dot           : lateral error rate of change (m/s), chuẩn hoá [-1, 1] theo E_LAT_DOT_MAX
  [2]  theta_e           : heading error (rad), chuẩn hoá [-1, 1] theo THETA_MAX
  [3]  curvature         : road curvature (1/m), chuẩn hoá [-1, 1] theo CURVATURE_MAX
  [4]  line_visible      : 1.0 nếu tracking vạch, 0.0 nếu mất vạch
  [5]  d_left            : khoảng cách zone trái (0°–36°), chuẩn hoá [0, 1]
  [6]  d_front_left      : khoảng cách zone trái-trước (36°–72°)
  [7]  d_front           : khoảng cách zone trước (72°–108°), quan trọng nhất
  [8]  d_front_right     : khoảng cách zone phải-trước (108°–144°)
  [9]  d_right           : khoảng cách zone phải (144°–180°)
  [10] side_diff         : chênh lệch khoảng trống (d_left - d_right) ∈ [-1, 1] (Dương = né trái, Âm = né phải)
  [11] min_front_dist    : khoảng cách gần nhất trong 3 zone phía trước [0, 1]
  [12] obstacle_detected : 1.0 nếu có vật cản trước mặt (d_front < OBSTACLE_THRESH), else 0.0
  [13] prev_steer        : góc lái của bước trước [-1, 1] (chống rung giật)
  [14] prev_throttle     : mức ga của bước trước [0, 1] (kiểm soát gia tốc)
"""

import math
import numpy as np

# =====================================================================
# CONSTANTS
# =====================================================================
STATE_DIM        = 15      # tổng số chiều của S_t

E_LAT_MAX        = 0.30    # (m) lateral error tối đa để normalize → [-1, 1]
E_LAT_DOT_MAX    = 0.60    # (m/s) tốc độ trôi ngang tối đa
THETA_MAX        = 0.60    # (rad) ≈ 34°, heading error tối đa
CURVATURE_MAX    = 2.0     # (1/m) độ cong tối đa (bán kính cua R = 0.5m)

LIDAR_OFFSET_DEG = 180.0   # góc offset gắn LiDAR của xe
LIDAR_MAX_RANGE  = 1.5     # (m) chuẩn hoá d: 1.5m → 1.0
OBSTACLE_THRESH  = 0.50    # (m) d_front < ngưỡng này → obstacle_detected = 1

# 5 zones, mỗi zone 36° — phủ kín 180° phía trước
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
    Trích xuất N_ZONES khoảng cách min từ LaserScan và các dẫn xuất không gian.

    Args:
        scan_msg: sensor_msgs/LaserScan hoặc None

    Returns:
        zones (np.ndarray shape (N_ZONES,)) — normalized [0, 1]
        side_diff (float)                   — (d_left - d_right) ∈ [-1, 1]
        min_front_dist (float)              — min(front-left, front, front-right) ∈ [0, 1]
        obstacle_detected (float)           — 1.0 hoặc 0.0
    """
    zones = np.ones(N_ZONES, dtype=np.float32)  # default = 1.0 (xa, an toàn)

    if scan_msg is None:
        return zones, 0.0, 1.0, 0.0

    angle_min_deg = math.degrees(scan_msg.angle_min)
    angle_inc_deg = math.degrees(scan_msg.angle_increment)
    zone_mins = [LIDAR_MAX_RANGE] * N_ZONES

    for i, raw_d in enumerate(scan_msg.ranges):
        if not (scan_msg.range_min < raw_d < scan_msg.range_max):
            continue
        d = min(raw_d, LIDAR_MAX_RANGE)

        raw_deg = angle_min_deg + i * angle_inc_deg
        deg = (raw_deg + LIDAR_OFFSET_DEG) % 360  # chuyển về frame của xe

        if not (0 <= deg <= 180):
            continue

        for z, (lo, hi) in enumerate(ZONE_ANGLES):
            if lo <= deg < hi:
                if d < zone_mins[z]:
                    zone_mins[z] = d
                break

    # Normalize zones [0, 1]
    zones = np.array([d / LIDAR_MAX_RANGE for d in zone_mins], dtype=np.float32)

    # Chênh lệch khoảng trống 2 sườn: d_left - d_right
    side_diff = float(np.clip(zones[0] - zones[4], -1.0, 1.0))

    # Khoảng cách tối thiểu trong hành lang phía trước (zones 1, 2, 3)
    min_front_dist = float(min(zones[1], zones[2], zones[3]))

    # Cờ chướng ngại vật trước mặt
    front_dist = zone_mins[2]
    obstacle_detected = 1.0 if front_dist < OBSTACLE_THRESH else 0.0

    return zones, side_diff, min_front_dist, obstacle_detected


def extract_state(lane_state, scan_msg, prev_steer=0.0, prev_throttle=0.0, prev_e_y=None, dt=0.033):
    """
    Xây dựng vector trạng thái hoàn chỉnh S_t (15 chiều).

    Args:
        lane_state: LaneState hoặc None
        scan_msg  : sensor_msgs/LaserScan hoặc None
        prev_steer: float [-1, 1]
        prev_throttle: float [0, 1]
        prev_e_y  : float (lateral error m ở frame trước, để tính e_y_dot) hoặc None
        dt        : float thời gian giữa 2 frame (s)

    Returns:
        state_vec (np.ndarray shape (STATE_DIM,), dtype=float32)
        info (dict) — debug info
    """
    # ---- 1. Vision Features ----
    if lane_state is not None:
        raw_e_y = float(lane_state.lateral_error_m)
        e_y = np.clip(raw_e_y / E_LAT_MAX, -1.0, 1.0)

        # Tính đạo hàm e_y_dot (tốc độ lệch tâm)
        if prev_e_y is not None and dt > 1e-4:
            raw_e_y_dot = (raw_e_y - prev_e_y) / dt
        else:
            raw_e_y_dot = 0.0
        e_y_dot = np.clip(raw_e_y_dot / E_LAT_DOT_MAX, -1.0, 1.0)

        raw_theta_e = float(lane_state.heading_error)
        theta_e = np.clip(raw_theta_e / THETA_MAX, -1.0, 1.0)

        raw_curv = float(getattr(lane_state, 'curvature', 0.0))
        curvature = np.clip(raw_curv / CURVATURE_MAX, -1.0, 1.0)

        from robot.estimation.lane_state import TrackingState
        ts = getattr(lane_state, 'tracking_state', TrackingState.SEARCH)
        line_visible = 1.0 if ts in (TrackingState.TRACKING, TrackingState.UNCERTAIN) else 0.0
    else:
        raw_e_y = 0.0
        e_y = 0.0
        raw_e_y_dot = 0.0
        e_y_dot = 0.0
        raw_theta_e = 0.0
        theta_e = 0.0
        raw_curv = 0.0
        curvature = 0.0
        line_visible = 0.0

    # ---- 2. LiDAR Features ----
    lidar_zones, side_diff, min_front_dist, obstacle_detected = extract_lidar_zones(scan_msg)

    # ---- 3. Action History ----
    p_steer = float(np.clip(prev_steer, -1.0, 1.0))
    p_throttle = float(np.clip(prev_throttle, 0.0, 1.0))

    # ---- 4. Ghép vector S_t (15 chiều) ----
    state_vec = np.array([
        e_y,
        e_y_dot,
        theta_e,
        curvature,
        line_visible,
        lidar_zones[0],
        lidar_zones[1],
        lidar_zones[2],
        lidar_zones[3],
        lidar_zones[4],
        side_diff,
        min_front_dist,
        obstacle_detected,
        p_steer,
        p_throttle,
    ], dtype=np.float32)

    info = {
        'e_y_raw'          : raw_e_y,
        'e_y_dot_raw'      : raw_e_y_dot,
        'theta_e_raw'      : raw_theta_e,
        'curvature_raw'    : raw_curv,
        'line_visible'     : line_visible,
        'lidar_zones_m'    : lidar_zones * LIDAR_MAX_RANGE,
        'side_diff'        : side_diff,
        'min_front_dist_m' : min_front_dist * LIDAR_MAX_RANGE,
        'obstacle_detected': obstacle_detected,
        'prev_steer'       : p_steer,
        'prev_throttle'    : p_throttle,
    }

    return state_vec, info
