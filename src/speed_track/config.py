#!/usr/bin/env python3
"""
V3 / V3.1 Lane Tracking — Centralized Configuration

All tunable parameters in one place. No magic numbers scattered in code.
Parameters marked [CALIBRATE] must be measured/tuned on the real car.
Parameters marked [TUNE] should be adjusted after initial testing.

V3.1.00 additions are grouped at the bottom and have safe defaults
that preserve V3.0 behavior when not explicitly changed.
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
    # [V3.1 Archive] roi_y_start: float = 0.30
    roi_y_start: float = 0.44   # [TUNE] New calibrated value
    roi_y_end: float = 1.00     # bottom of image

    # ================================================================
    # HSV COLOR SEGMENTATION — Red/orange lane markings
    # ================================================================
    # Two hue ranges for wrap-around (red sits at both ends of hue spectrum)
    hsv_h1_min: int = 0        
    hsv_h1_max: int = 179        
    hsv_h2_min: int = 39        
    hsv_h2_max: int = 179       
    
    # Optional upper bounds for S and V (usually 255, but tunable)
    hsv_s_max: int = 255        
    hsv_v_max: int = 255        
    
    # Near zone (strict - rejects noise)
    # [V3.1 Archive] hsv_s_min = 100, hsv_v_min = 105
    hsv_s_min: int = 50        
    hsv_v_min: int = 118        
    
    # Far zone (loose - catches distant faded lines)
    # [V3.1 Archive] hsv_s_min_far = 50, hsv_v_min_far = 80
    hsv_s_min_far: int = 50     
    hsv_v_min_far: int = 117     
    hsv_far_y_split: float = 0.55 # Top 55% of image (Y=0 to 264) uses FAR filter (Covers Horizon Scanner 200-260)
    
    # LAB Constraint (Optional) to reject non-red colors that pass HSV
    use_lab_constraint: bool = False   
    lab_a_min: int = 0        

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
    # [V3.1 Archive] use_lab_constraint: bool = False, lab_a_min: int = 135
    use_lab_constraint: bool = False   
    lab_a_min: int = 0        # [TUNE] New calibrated value

    # ================================================================
    # MORPHOLOGY — Light filtering only
    # ================================================================
    # OPEN only (erode then dilate) — removes noise dots
    # Do NOT use CLOSE — it connects the dashed center line
    # [V3.1 Archive] morph_kernel_size: int = 3
    morph_kernel_size: int = 5        
    morph_iterations: int = 1

    # ================================================================
    # BEV / IPM TRANSFORM
    # ================================================================
    # Source points on the original image (trapezoid on road surface)
    # Order: bottom-left, bottom-right, top-right, top-left
    # [CALIBRATE] using calib_bev.py on the real car
    # BEV Source points (Perspective) - Extended Horizon for Predictive Speed
    bev_src_pts: np.ndarray = field(default_factory=lambda: np.float32([
        [0, 480],       # bottom-left
        [640, 480],     # bottom-right
        [518, 260],     # top-right (Extended from 300 to 260)
        [122, 260]      # top-left (Extended from 300 to 260)
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
    max_steer_angle_rad: float = 0.8       # [TUNE] TĂNG số này lên để vô lăng bẻ êm hơn (không bị max 1.0 ngay lập tức)

    # ================================================================
    # STANLEY CONTROLLER (V2 — after PP is stable)
    # ================================================================
    stanley_k: float = 0.3                 # [TUNE]
    stanley_enabled: bool = False          # DO NOT enable until PP is validated

    # ================================================================
    # STEERING FILTER
    # ================================================================
    max_steer_rate: float = 1.0           # [TUNE] Trả lại 1.0: Không giới hạn tốc độ bẻ lái để vào cua gắt
    steer_lpf_alpha: float = 1.0          # [TUNE] Trả lại 1.0: Phản hồi vô lăng tức thì, không bị trễ
    high_speed_steer_gain: float = 1.0    # [V3.1] gain tại max_speed (1.0 = no change, 0.65 = reduce 35%)

    # ================================================================
    # SPEED CONTROL
    # ================================================================
    max_speed: float = 0.50                # [TUNE] throttle
    min_speed: float = 0.18                # [TUNE]
    cruise_speed: float = 0.40             # [TUNE] default straight-line speed
    a_lat_max: float = 2.0                 # [TUNE] lateral accel limit (m/s²)
    speed_confidence_thresh: float = 0.5   # reduce speed below this confidence
    speed_to_throttle_factor: float = 1.0  # [CALIBRATE] TĂNG HỆ SỐ NÀY ĐỂ XE BỐC HƠN (v(m/s) → throttle)

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
    record_video: bool = False             # [TUNE] TẮT QUAY VIDEO ĐỂ XE CHẠY KHÔNG BỊ DELAY (Tăng FPS)
    video_fps: int = 5
    log_csv: bool = True
    loop_rate: int = 20                    # Hz
    debug_mode: bool = True                # [V3.1] True = render visualizer, False = skip (faster)

    # ================================================================
    # ROS TOPICS
    # ================================================================
    camera_topic: str = '/csi_cam_0/image_raw'
    lidar_topic: str = '/scan'
    node_name: str = 'speed_racing_v3'

    # ================================================================
    # V3.1 — PERFORMANCE OPTIMIZATIONS
    # ================================================================
    # Processing resolution scale (1.0 = full 640x480, 0.5 = 320x240)
    # Lower = faster FPS but less pixel data for lane detection
    processing_scale: float = 1.0          # [V3.1] Safe default = no change

    # Curvature history for oval stability bonus
    curvature_history_size: int = 10       # [V3.1] number of frames to track
    curvature_stability_bonus: float = 1.15  # [V3.1] +15% speed when curvature stable
    curvature_stability_thresh: float = 0.1  # [V3.1] std threshold for "stable"

    # Area heuristic V2
    area_k: float = 0.15                   # [V3.1] area correction gain (V3 default)
    area_deadband: float = 0.0             # [V3.1] ignore area_ratio below this (V3 = 0)

    # Early Horizon Scanner
    horizon_warning_enabled: bool = True   # [V3.1] Enable scanning above BEV for curves
    # [V3.1 Archive] horizon_scan_y_start = 200, horizon_scan_y_end = 280
    horizon_scan_y_start: int = 219        
    horizon_scan_y_end: int = 379        
    horizon_angle_thresh: float = 18.00   # trigger brake if line angle > 15 degrees
    horizon_center_zone: int = 147        # only consider lines within +-71px of center
    horizon_pix_thresh: int = 50        # min pixels to consider a valid line
