#!/usr/bin/env python3
"""
V3 Lane Tracking — Centralized Configuration

All tunable parameters in one place. No magic numbers scattered in code.
Parameters marked [CALIBRATE] must be measured/tuned on the real car.
Parameters marked [TUNE] should be adjusted after initial testing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class V3Config:
    """Complete configuration for the V3 lane tracking pipeline."""

    # ================================================================
    # CAMERA
    # ================================================================
    image_width: int = 640
    image_height: int = 480

    # Camera intrinsics (from checkerboard calibration)
    # Set to None to skip undistortion
    camera_matrix: Optional[np.ndarray] = None      # [CALIBRATE]
    dist_coeffs: Optional[np.ndarray] = None         # [CALIBRATE]

    # ================================================================
    # ROI — One wide region of interest
    # ================================================================
    # Fraction of image height: crop [roi_y_start * H : roi_y_end * H]
    # Wide enough that lane markings aren't clipped during curves
    roi_y_start: float = 0.30   # [TUNE] skip top 30% (sky / far noise)
    roi_y_end: float = 1.00     # bottom of image

    # ================================================================
    # HSV COLOR SEGMENTATION — Red/orange lane markings
    # ================================================================
    # Two hue ranges for wrap-around (red sits at both ends of hue spectrum: 0-10 and 170-179)
    hsv_h1_min: int = 0         
    hsv_h1_max: int = 10        
    hsv_h2_min: int = 170       
    hsv_h2_max: int = 179       
    hsv_s_min: int = 100        
    hsv_v_min: int = 100        

    # Use CLAHE preprocessing for lighting robustness
    use_clahe: bool = False #nếu dùng cân bằng ánh sáng cục bộ thì True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Lightweight white-balance normalization (gray-world)
    # Stabilizes hue under indoor fluorescent lighting
    # [Tắt tạm thời vì video MP4 bị sai màu]
    use_white_balance: bool = False

    # LAB chromaticity constraint (rejects gray/white reflections)
    # In OpenCV LAB: a=128 is neutral, a>128 is red direction.
    # Red/orange markings have a > ~135; gray floor reflections have a ≈ 128.
    # [Tắt tạm thời vì video MP4 có thể làm mất kênh màu LAB]
    use_lab_constraint: bool = False
    lab_a_min: int = 135        # [TUNE] minimum LAB a-channel for red/orange

    # ================================================================
    # MORPHOLOGY — Light filtering only
    # ================================================================
    # OPEN only (erode then dilate) — removes noise dots
    # Do NOT use CLOSE — it connects the dashed center line
    morph_kernel_size: int = 3
    morph_iterations: int = 1

    # ================================================================
    # BEV / IPM TRANSFORM
    # ================================================================
    # Source points on the original image (trapezoid on road surface)
    # Order: bottom-left, bottom-right, top-right, top-left
    # [CALIBRATE] using calib_bev.py on the real car
    # Adjusted for the low-mounted, downward-angled camera:
    # - Mở rộng tối đa đáy (0 -> 640) để không cắt mất vạch sát xe.
    # - Hạ thấp đỉnh BEV xuống y=300 để né vùng lóa sáng phản chiếu mặt sàn.
    bev_src_pts: np.ndarray = field(default_factory=lambda: np.float32([
        [0,   480],   # bottom-left (mở rộng tối đa)
        [640, 480],   # bottom-right (mở rộng tối đa)
        [540, 300],   # top-right (hạ xuống y=300 để né vùng lóa)
        [100, 300],   # top-left (hạ xuống y=300 để né vùng lóa)
    ]))

    # Destination rectangle in BEV space
    bev_dst_pts: np.ndarray = field(default_factory=lambda: np.float32([
        [640 * 0.20, 480],   # bottom-left (wider than V2's 0.3)
        [640 * 0.80, 480],   # bottom-right
        [640 * 0.80, 0],     # top-right
        [640 * 0.20, 0],     # top-left
    ]))

    # Metric calibration: how many pixels per meter in BEV space
    # [CALIBRATE] measure a known distance on the track in BEV
    # Sửa lại scale cho đúng với xe mô hình (RC car): 1 pixel = 1 mm
    # Track rộng ~400 pixels trong BEV => tương đương 0.4 mét (đúng với expected_lane_width_m)
    px_per_meter_x: float = 1000.0   # [CALIBRATE] horizontal
    px_per_meter_y: float = 1000.0   # [CALIBRATE] vertical

    # ================================================================
    # SLIDING WINDOW LANE DETECTION
    # ================================================================
    sw_n_windows: int = 12          # number of sliding windows per line
    sw_margin: int = 50             # half-width of window (pixels)
    sw_min_pix: int = 30            # minimum pixels to recenter window
    sw_min_peak_height: int = 50    # minimum histogram peak height (increased to avoid noise specks)
    sw_min_peak_distance: int = 60  # minimum distance between peaks (pixels)

    # ================================================================
    # RANSAC POLYNOMIAL FITTING
    # ================================================================
    poly_degree: int = 2
    ransac_residual_threshold: float = 5.0    # pixels
    ransac_max_trials: int = 50 # có thể tăng lên <= 100
    ransac_min_samples: int = 10

    # ================================================================
    # CONFIDENCE SCORING
    # ================================================================
    min_inlier_count: int = 150     # below this → confidence ≈ 0 (increased from 50)
    expected_inlier_count: int = 400  # above this → count component ≈ 1 (increased from 300)
    max_fit_rmse: float = 8.0       # pixels; above this → rmse component ≈ 0
    conf_weight_count: float = 0.6  # Give more weight to pixel count
    conf_weight_rmse: float = 0.2
    conf_weight_inlier_ratio: float = 0.2

    # ================================================================
    # LANE GEOMETRY
    # ================================================================
    # Physical lane width (distance between the two boundary lines)
    # Track thực tế của xe mô hình thường có độ rộng toàn dải (trái sang phải) khoảng 60cm
    expected_lane_width_m: float = 0.60    # [CALIBRATE] meters
    lane_width_tolerance: float = 0.35     # ± 35% of expected width

    # ================================================================
    # TEMPORAL FILTER (alpha-beta / EMA)
    # ================================================================
    alpha_position: float = 1.0     # [TUNE] EMA for centerline position (1.0 = no delay)
    alpha_heading: float = 1.0      # [TUNE]
    alpha_curvature: float = 1.0    # [TUNE]
    alpha_width: float = 1.0        # [TUNE]
    confidence_decay: float = 0.95  # per-frame decay when no measurement

    # ================================================================
    # MEASUREMENT GATING
    # ================================================================
    max_lateral_jump_m: float = 0.40       # [TUNE] meters (increased to prevent false rejections)
    max_curvature_jump: float = 2.0        # [TUNE] 1/m (increased to allow RANSAC wiggles)
    min_confidence_gate: float = 0.15      # reject observations below this

    # ================================================================
    # TRAJECTORY — Look-ahead
    # ================================================================
    n_lookahead_points: int = 10
    lookahead_L0: float = 0.20      # [TUNE] base lookahead distance (m)
    lookahead_kv: float = 0.5       # [TUNE] speed gain (m per m/s)
    lookahead_kk: float = 0.1       # [TUNE] curvature gain (m per 1/m)
    lookahead_Lmin: float = 0.15    # minimum lookahead (m)
    lookahead_Lmax: float = 0.60    # maximum lookahead (m)

    # ================================================================
    # PURE PURSUIT CONTROLLER (V1 — baseline)
    # ================================================================
    wheelbase: float = 0.16                # [CALIBRATE] meters (axle-to-axle)
    max_steer_angle_rad: float = 0.45      # [CALIBRATE] Giảm số này = TĂNG lực bẻ lái của servo

    # ================================================================
    # STANLEY CONTROLLER (V2 — after PP is stable)
    # ================================================================
    stanley_k: float = 0.3                 # [TUNE]
    stanley_enabled: bool = False          # DO NOT enable until PP is validated

    # ================================================================
    # STEERING FILTER
    # ================================================================
    max_steer_rate: float = 1.0           # [TUNE] max change per frame (cho phép servo xoay gắt hơn)
    steer_lpf_alpha: float = 1.0          # [TUNE] 1.0 = no filter (phản hồi vô lăng tức thì)

    # ================================================================
    # SPEED CONTROL
    # ================================================================
    max_speed: float = 0.55                # [TUNE] throttle
    min_speed: float = 0.25                # [TUNE]
    cruise_speed: float = 0.45             # [TUNE] default straight-line speed
    a_lat_max: float = 2.0                 # [TUNE] lateral accel limit (m/s²)
    speed_confidence_thresh: float = 0.5   # reduce speed below this confidence
    speed_to_throttle_factor: float = 1.3  # [CALIBRATE] TĂNG HỆ SỐ NÀY ĐỂ XE BỐC HƠN (v(m/s) → throttle)

    # Speed PID (for encoder feedback)
    speed_pid_kp: float = 0.5
    speed_pid_ki: float = 0.0
    speed_pid_kd: float = 0.1
    use_encoder: bool = False              # [CALIBRATE] set True if encoder available

    # ================================================================
    # STATE MACHINE TIMEOUTS
    # ================================================================
    uncertain_timeout: float = 0.5         # seconds before PREDICTING
    predicting_timeout: float = 2.0        # seconds before RECOVERY
    recovery_timeout: float = 3.0          # seconds before E_STOP
    search_timeout: float = 999.0          # seconds in SEARCH before E_STOP

    # State transition thresholds
    tracking_confidence_min: float = 0.5   # need this to enter/stay TRACKING
    uncertain_confidence_thresh: float = 0.4  # drop below → UNCERTAIN

    # ================================================================
    # STEERING INVERSION
    # ================================================================
    # V2 had: self.racer.steer(-steer, throttle) — servo is inverted
    steer_invert: bool = False

    # ================================================================
    # DEBUG & LOGGING
    # ================================================================
    record_video: bool = False             # Set False for max FPS during live racing
    video_fps: int = 30
    log_csv: bool = True
    loop_rate: int = 40                    # Hz (Boosted to 40Hz for low latency)

    # ================================================================
    # ROS TOPICS
    # ================================================================
    camera_topic: str = '/csi_cam_0/image_raw'
    lidar_topic: str = '/scan'
    node_name: str = 'speed_racing_v3'
