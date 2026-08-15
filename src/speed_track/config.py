#!/usr/bin/env python3
"""
V3 Lane Tracking — Centralized Configuration

All tunable parameters in one place. No magic numbers scattered in code.
Parameters marked [CALIBRATE] must be measured/tuned on the real car.
Parameters marked [TUNE] should be adjusted after initial testing.
"""

import math
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
    hsv_s_min: int = 60         # [TUNE] Lowered to 60 to capture red lines under shadows
    hsv_v_min: int = 60         # [TUNE] Lowered to 60 for low-light robustness

    # Use CLAHE preprocessing for lighting robustness
    use_clahe: bool = True #nếu dùng cân bằng ánh sáng cục bộ thì True
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
    # [CALIBRATE] BEV destination rectangle width = 640 * (0.80 - 0.20) = 384 px.
    # For a track width of 0.60m: px_per_meter = 384 px / 0.60 m = 640.0 px/m.
    px_per_meter_x: float = 640.0   # [CALIBRATE] horizontal scale (384 px / 0.60 m)
    px_per_meter_y: float = 640.0   # [CALIBRATE] vertical scale

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
    min_inlier_count: int = 150             # below this → confidence ≈ 0 (for solid lines)
    expected_inlier_count: int = 400        # above this → count component ≈ 1 (for solid lines)
    min_inlier_count_dashed: int = 60       # below this → confidence ≈ 0 (for dashed lines)
    expected_inlier_count_dashed: int = 180 # above this → count component ≈ 1 (for dashed lines)
    max_fit_rmse: float = 8.0               # pixels; above this → rmse component ≈ 0
    conf_weight_count: float = 0.6          # Give more weight to pixel count
    conf_weight_rmse: float = 0.2
    conf_weight_inlier_ratio: float = 0.2

    # ================================================================
    # LANE GEOMETRY
    # ================================================================
    # Physical lane width (distance between the two boundary lines)
    # Track thực tế của xe mô hình thường có độ rộng toàn dải (trái sang phải) khoảng 60cm
    expected_lane_width_m: float = 0.60    # [CALIBRATE] meters
    lane_width_tolerance: float = 0.20     # ± 20% of expected width (tightened from 35%)
    
    # Startup calibration guard
    startup_check_frames: int = 20         # Number of dual-line frames to validate calibration
    startup_timeout_frames: int = 60       # Timeout frames (at ~30fps) to abort if dual-lines never detected
    startup_width_tolerance: float = 0.18  # Max deviation allowed during startup validation

    # ================================================================
    # TEMPORAL FILTER (alpha-beta / EMA) & DEAD RECKONING
    # ================================================================
    alpha_position: float = 0.6     # [TUNE] EMA for centerline position (1.0 = no delay)
    alpha_heading: float = 1.0      # [TUNE]
    alpha_curvature: float = 1.0    # [TUNE]
    alpha_width: float = 1.0        # [TUNE]
    confidence_decay: float = 0.95  # per-frame decay when no measurement
    
    # Short-term curved dead reckoning when lines are lost in curves
    predict_hold_duration: float = 0.30    # seconds to keep 100% curvature & heading before decay
    reacquisition_gate_m: float = 0.25     # max lateral jump allowed when re-detecting line

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
    # UNIFIED ACKERMANN CHASSIS & KINEMATIC MODEL
    # ================================================================
    wheelbase_m: float = 0.14              # [CALIBRATE] Physical wheelbase L (m)
    max_steer_rad: float = 0.436           # Max front wheel steering angle δ_max (rad) ~ 25 deg (+right, -left)
    max_steer_rate_rad_s: float = 2.5      # Servo slew rate limit (rad/s)
    car_length_m: float = 0.25             # [CALIBRATE] Total physical vehicle length (m)
    car_width_m: float = 0.18              # [CALIBRATE] Total physical vehicle width with tires (m)
    track_width_m: float = 0.60            # [CALIBRATE] Total width between outer red boundaries (m)
    safety_margin_m: float = 0.04          # [TUNE] Distance buffer from outer red line (m)
    max_evade_offset_m: float = 0.18       # [TUNE] Clamped max virtual offset (m)
    offset_ramp_rate: float = 0.60         # [TUNE] Virtual offset ramp speed (m/s)

    # ================================================================
    # PURE PURSUIT & STANLEY CONTROLLERS
    # ================================================================
    stanley_k: float = 0.3                 # [TUNE]
    stanley_enabled: bool = False          # DO NOT enable until PP is validated

    # ================================================================
    # STEERING FILTER
    # ================================================================
    steer_lpf_alpha: float = 0.8          # [TUNE] 1.0 = no filter (instant response)

    # ================================================================
    # SPEED CONTROL
    # ================================================================
    cruise_speed: float = 0.25             # m/s (an toàn trên sa bàn)
    max_speed: float = 0.35                # m/s (giới hạn an toàn trên đường thẳng)
    crawl_speed: float = 0.12              # m/s (chế độ bò khi mất target)
    min_speed: float = 0.15                # m/s (tốc độ tối thiểu)
    a_lat_max: float = 2.0                 # m/s² (giới hạn gia tốc ngang)
    curve_speed_min: float = 0.15          # m/s (tốc độ tối thiểu trong cua gắt)
    curve_slowdown_factor: float = 0.6     # hệ số giảm tốc khi cua gắt
    confidence_speed_weight: float = 0.4   # giảm tốc khi confidence thấp
    speed_confidence_thresh: float = 0.5   # reduce speed below this confidence
    speed_to_throttle_factor: float = 1.0  # tỉ lệ chuyển m/s -> throttle
    use_encoder: bool = False              # [CALIBRATE] set True if encoder available
    speed_pid_kp: float = 0.5
    speed_pid_ki: float = 0.0
    speed_pid_kd: float = 0.1
    steer_pid_kp: float = 1.8
    steer_pid_ki: float = 0.0
    steer_pid_kd: float = 0.08

    # ================================================================
    # APF OBSTACLE AVOIDANCE & FLANK CHECK
    # ================================================================
    lidar_offset_deg: float = 180.0        # [CALIBRATE] LiDAR mount rotation offset
    apf_gain: float = 0.15                 # [TUNE] repulsive force gain
    apf_influence_dist: float = 0.80       # [TUNE] APF influence radius (m)
    apf_frontal_bias: float = 0.3          # [TUNE] lateral bias for frontal obstacles
    obstacle_trigger_dist: float = 0.60    # [TUNE] start EVADING when closer than this (m)
    obstacle_clear_dist: float = 0.80      # [TUNE] obstacle passed threshold (m)
    obstacle_e_stop_dist: float = 0.20     # [TUNE] emergency stop distance (m)
    evade_steer_weight: float = 0.6        # [TUNE] APF weight during EVADING (0=lane only, 1=APF only)
    return_lateral_threshold: float = 0.05 # [TUNE] lateral error to consider "back on line" (m)
    evade_speed_factor: float = 0.6        # [TUNE] throttle multiplier during evasion

    # Flank / Side sector clearance check (prevents rear-wheel sideswiping)
    side_scan_start_deg: float = 70.0      # [TUNE] Side scan start angle (deg)
    side_scan_end_deg: float = 110.0       # [TUNE] Side scan end angle (deg)
    side_clear_dist: float = 0.35          # [TUNE] Side clearance threshold before returning (m)

    # ================================================================
    # CONTROL-LATTICE BICYCLE ROLLOUT PLANNER & SAFETY
    # ================================================================
    n_candidate_rollouts: int = 11         # Number of candidate bicycle rollouts spanning [-delta_max, +delta_max]
    n_candidate_trajectories: int = 11     # Alias for backwards compatibility
    rollout_horizon_s: float = 0.90        # Rollout forward planning time horizon (s)
    rollout_dt_s: float = 0.03             # Numerical integration timestep (s)

    # Cost Weights for Optimal Rollout Selection
    w_lane: float = 6.0                    # Weight for following the reference lane target
    w_steer: float = 0.05                  # Weight penalizing large steering angles
    w_rate: float = 0.05                   # Weight penalizing steering changes from current steer
    w_clearance: float = 1.20              # Weight penalizing proximity to obstacles
    w_progress: float = 0.10               # Weight rewarding longitudinal progress

    lidar_x_offset_m: float = 0.0          # [CALIBRATE] LiDAR mount X offset from vehicle centerline (m)
    lidar_y_offset_m: float = 0.10         # [CALIBRATE] LiDAR mount Y offset forward from rear axle (m)
    lidar_max_age_s: float = 0.10          # Max acceptable age of LaserScan data (s)
    lidar_max_sync_skew_s: float = 0.10    # Max time skew between Camera frame and LaserScan (s)
    lidar_timeout_s: float = 0.80          # Safe stop if no valid LiDAR scan received for this long (s)
    camera_timeout_s: float = 0.25         # Safe stop if no new camera frame received within this time (s)
    boundary_stale_timeout_s: float = 0.40 # Limit evasion offset if boundary not seen within this time (s)
    boundary_stop_timeout_s: float = 0.80  # Safe stop if both boundaries remain stale for this long (s)
    a_brake_max: float = 2.5               # Max deceleration for stopping distance check (m/s²)
    t_reaction_s: float = 0.10             # System reaction + brake lag time (s)

    @property
    def max_trajectory_curvature(self) -> float:
        """Maximum physically achievable path curvature: kappa_max = tan(delta_max) / L."""
        return math.tan(self.max_steer_rad) / self.wheelbase_m

    # ================================================================
    # STATE MACHINE TIMEOUTS
    # ================================================================
    uncertain_timeout: float = 0.30        # seconds before PREDICTING
    predicting_timeout: float = 0.40       # seconds in PREDICTING before RECOVERY
    recovery_timeout: float = 0.80         # seconds in RECOVERY before safe E_STOP
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
