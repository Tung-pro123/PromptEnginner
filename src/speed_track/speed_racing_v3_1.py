#!/usr/bin/env python3
"""
Speed Track Racing V3.1.00 — Performance-Optimized V3

Inherits the FULL V3 pipeline architecture:
  Camera → Undistort → ROI → HSV → Morph → BEV
  → Histogram + Sliding Window → RANSAC per line
  → Geometry Validation → Center Reconstruction
  → Temporal EMA Filter → Look-ahead Trajectory
  → Pure Pursuit / Stanley → Steering Filter → Speed Controller

V3.1 optimizations for higher speed + FPS (especially on oval tracks):
  1. Downscaled processing (320x240) — ~2-3x FPS boost
  2. Skip visualizer in ROS mode — saves ~3-5ms/frame
  3. Skip frame copy for ROI — saves ~1ms/frame
  4. Speed-adaptive steering — reduce sensitivity at high speed
  5. Curvature-history speed control — oval stability bonus
  6. Area heuristic v2 — deadband + speed-dependent gain
  7. Reduced RANSAC/sliding window iterations

All V3 advantages preserved:
  ✓ Multi-lane detection (L/C/R)
  ✓ RANSAC polynomial fitting
  ✓ Temporal EMA state estimation
  ✓ Measurement gating
  ✓ State machine (SEARCH/TRACKING/UNCERTAIN/etc.)
  ✓ Pure Pursuit + Stanley controllers
  ✓ CSV logging + optional video recording

Run modes:
  - ROS mode (on JetRacer): `rosrun speed_track speed_racing_v3_1.py`
  - Offline mode (laptop):    `python speed_racing_v3_1.py`
  - Offline video replay:     `python speed_racing_v3_1.py --video path/to/video.avi`
"""

import sys
sys.path.append("../../")

import os
import cv2
import math
import time
import argparse
import numpy as np

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

try:
    import rospy
    from sensor_msgs.msg import Image, LaserScan
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("WARNING: ROS not found. Running in offline mode.")

try:
    from src.core.control.racer_controller import RacerController
except ImportError:
    print("WARNING: RacerController not found. Using mock.")
    class RacerController:
        def steer(self, angle, speed): pass
        def stop(self): pass

# V3 modules (unchanged — all V3 advantages preserved)
from src.speed_track.config import V3Config
from src.speed_track.perception.undistort import Undistorter
from src.speed_track.perception.segmentation import ColorSegmenter
from src.speed_track.perception.bev import BEVTransform
from src.speed_track.perception.lane_detector import MultiLaneDetector, LaneDetectionResult, LineDetection
from src.speed_track.estimation.geometry import LaneGeometry
from src.speed_track.estimation.lane_state import LaneStateEstimator, TrackingState, LaneState
from src.speed_track.control.trajectory import TrajectoryGenerator, TrajectoryResult
from src.speed_track.control.pure_pursuit import PurePursuitController
from src.speed_track.control.stanley import StanleyController
from src.speed_track.control.steering_filter import SteeringFilter
from src.speed_track.control.speed_controller import SpeedController
from src.speed_track.debug.visualizer import DebugVisualizer
from src.speed_track.debug.logger import V3Logger


# ======================================================================
# V3.1 CONFIG PRESET — Aggressive tuning for oval speed
# ======================================================================

def make_v31_config():
    """Create a V3Config.
    
    [V3.1 Update] All performance optimizations, FPS settings, and speed boosts
    have been permanently moved to config.py to act as a single source of truth.
    This function now simply returns the default config so Live Calibrator 
    and other tools can modify it directly.
    """
    cfg = V3Config()
    return cfg


class SpeedRacingV31:
    """V3.1 runner — V3 pipeline with performance optimizations."""

    def __init__(self, config=None, video_path=None):
        """
        Args:
            config: V3Config instance. Uses V3.1 preset if None.
            video_path: Path to a video file for offline replay. None for live camera.
        """
        self.cfg = config or make_v31_config()
        self.video_path = video_path

        # ---- V3.1: Compute processing dimensions ----
        self.proc_w = int(self.cfg.image_width * self.cfg.processing_scale)
        self.proc_h = int(self.cfg.image_height * self.cfg.processing_scale)
        self.use_downscale = (self.cfg.processing_scale < 1.0)

        # ---- V3.1: Create a scaled config for modules that depend on image size ----
        # We need modules to work at the processing resolution
        if self.use_downscale:
            self.proc_cfg = self._make_scaled_config()
        else:
            self.proc_cfg = self.cfg

        # ---- Initialize pipeline modules (use proc_cfg for processing) ----
        self.undistorter = Undistorter(self.cfg)  # Undistort at full res
        self.segmenter = ColorSegmenter(self.proc_cfg)
        self.bev = BEVTransform(self.proc_cfg)
        self.lane_detector = MultiLaneDetector(self.proc_cfg)
        self.geometry = LaneGeometry(self.proc_cfg, self.bev)
        self.state_estimator = LaneStateEstimator(self.proc_cfg, self.bev)
        self.trajectory_gen = TrajectoryGenerator(self.proc_cfg, self.bev)
        self.pp_controller = PurePursuitController(self.cfg)
        self.stanley_controller = StanleyController(self.cfg)
        self.steering_filter = SteeringFilter(self.cfg)
        self.speed_controller = SpeedController(self.cfg)
        self.visualizer = DebugVisualizer(self.cfg) if self.cfg.debug_mode else None

        # ---- Hardware ----
        self.racer = RacerController()
        self.racer.stop()

        # ---- ROS ----
        self.latest_image = None
        self.latest_scan = None

        if HAS_ROS and video_path is None:
            rospy.init_node('speed_racing_v3_1', anonymous=True)
            rospy.Subscriber(self.cfg.camera_topic, Image, self._cam_cb)
            rospy.Subscriber(self.cfg.lidar_topic, LaserScan, self._lidar_cb)

        # ---- Logging ----
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        self.logger = V3Logger(log_dir, prefix='v31') if self.cfg.log_csv else None
        self.video_writer = None

        if self.cfg.record_video:
            ts = time.strftime('%Y%m%d_%H%M%S')
            vid_path = os.path.join(log_dir, f'v31_{ts}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(
                vid_path, fourcc, self.cfg.video_fps,
                (self.cfg.image_width * 3, self.cfg.image_height)  # 3 panels: Raw, HSV, BEV
            )
            print(f"Recording video to: {vid_path}")

        # ---- State ----
        self.current_speed = 0.0    # Estimated or measured speed (m/s)
        self.current_throttle = 0.0  # V3.1: Track current throttle for speed estimation
        self.frame_count = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0
        self.prev_horizon_state = "UNKNOWN"

        print(f"=== V3.1.00 initialized ===")
        print(f"  Processing: {self.proc_w}x{self.proc_h} (scale={self.cfg.processing_scale})")
        print(f"  Speed: {self.cfg.min_speed} - {self.cfg.max_speed} (cruise={self.cfg.cruise_speed})")
        print(f"  Steer gain@max: {self.cfg.high_speed_steer_gain}, LPF={self.cfg.steer_lpf_alpha}")
        print(f"  Stability bonus: {self.cfg.curvature_stability_bonus}x")
        print(f"  Debug mode: {self.cfg.debug_mode}")

    def _make_scaled_config(self):
        """Create a copy of config with image dimensions and BEV points scaled down.

        This ensures all pixel-based parameters work correctly at reduced resolution.
        Metric parameters (px_per_meter) are also scaled so physical measurements
        remain accurate.
        """
        import copy
        pcfg = copy.deepcopy(self.cfg)
        s = self.cfg.processing_scale

        pcfg.image_width = self.proc_w
        pcfg.image_height = self.proc_h

        # Scale BEV source and destination points
        pcfg.bev_src_pts = self.cfg.bev_src_pts * s
        pcfg.bev_dst_pts = self.cfg.bev_dst_pts * s

        # Scale metric calibration (fewer pixels per meter at lower res)
        pcfg.px_per_meter_x = self.cfg.px_per_meter_x * s
        pcfg.px_per_meter_y = self.cfg.px_per_meter_y * s

        # Scale sliding window parameters
        pcfg.sw_margin = max(15, int(self.cfg.sw_margin * s))
        pcfg.sw_min_pix = max(10, int(self.cfg.sw_min_pix * s))
        pcfg.sw_min_peak_height = max(15, int(self.cfg.sw_min_peak_height * s))
        pcfg.sw_min_peak_distance = max(20, int(self.cfg.sw_min_peak_distance * s))

        # Scale confidence thresholds for lower pixel count
        pcfg.min_inlier_count = max(30, int(self.cfg.min_inlier_count * s * s))
        pcfg.expected_inlier_count = max(80, int(self.cfg.expected_inlier_count * s * s))

        return pcfg

    # ==================================================================
    # ROS CALLBACKS (identical to V3)
    # ==================================================================

    def _cam_cb(self, msg):
        """ROS camera callback — converts Image msg to BGR numpy array."""
        try:
            if 'compressed' in msg.encoding:
                self.latest_image = cv2.imdecode(
                    np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1)
                if 'rgb' in msg.encoding:
                    self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    self.latest_image = img
        except Exception as e:
            if HAS_ROS:
                rospy.logerr_throttle(5, f"Camera callback error: {e}")

    def _lidar_cb(self, msg):
        """ROS LiDAR callback — stores latest scan for future use."""
        self.latest_scan = msg

    # ==================================================================
    # CORE PIPELINE — V3.1 optimized
    # ==================================================================

    def process_frame(self, frame):
        """Run the V3.1 pipeline on a single frame.

        Same output as V3.process_frame but optimized for speed:
        - Downscaled processing
        - Conditional visualizer
        - Speed-adaptive steering
        - Curvature-history speed control

        Args:
            frame: BGR image (640x480).

        Returns:
            (steer_filtered, throttle, dashboard_image, lane_state)
        """
        cfg = self.cfg
        pcfg = self.proc_cfg

        # Ensure correct size at full resolution
        if frame.shape[:2] != (cfg.image_height, cfg.image_width):
            frame = cv2.resize(frame, (cfg.image_width, cfg.image_height))

        # ---- 1. Undistortion (at full res, skipped if no calibration) ----
        frame_undist = self.undistorter.process(frame)

        # ---- V3.1: Downscale for processing ----
        if self.use_downscale:
            proc_frame = cv2.resize(frame_undist, (self.proc_w, self.proc_h),
                                     interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame_undist

        # ---- 2. ROI crop ----
        # V3.1: No .copy() — process in-place (save ~1ms)
        roi_y_start = int(pcfg.roi_y_start * pcfg.image_height)
        proc_frame[:roi_y_start, :] = 0

        # ---- 3. Color segmentation ----
        mask = self.segmenter.process(proc_frame)

        # ---- 3.5. Horizon Scanner (Cơ chế mới: Định nghĩa STRAIGHT bằng sự hiện diện của vạch ở xa) ----
        horizon_state = "UNKNOWN"
        if getattr(cfg, 'horizon_warning_enabled', False):
            # Lấy tọa độ gốc chưa scale để map 1:1 với Calibrator.
            h_start_raw = getattr(cfg, 'horizon_scan_y_start', 152)
            h_end_raw = getattr(cfg, 'horizon_scan_y_end', 234)
            
            # Tọa độ cắt trên mask (đã áp dụng processing_scale)
            h_start_proc = int(h_start_raw * cfg.processing_scale)
            h_end_proc = int(h_end_raw * cfg.processing_scale)
            
            h_start_proc = max(0, min(h_start_proc, mask.shape[0]))
            h_end_proc = max(h_start_proc, min(h_end_proc, mask.shape[0]))
            
            # Tọa độ trên frame gốc để vẽ
            h_start = max(0, min(h_start_raw, frame.shape[0]))
            h_end = max(h_start, min(h_end_raw, frame.shape[0]))
            
            if h_end_proc > h_start_proc:
                # Tận dụng luôn mask đã được lọc màu ở bước 3, không cần HSV lại!
                h_mask_proc = mask[h_start_proc:h_end_proc, :].copy()
                
                # Tính số lượng điểm ảnh (Dùng non-zero thay vì moments)
                pixel_count_proc = cv2.countNonZero(h_mask_proc)
                
                # Đưa pixel count về tỷ lệ gốc để không bị sai với config GUI
                pixel_count = pixel_count_proc / (cfg.processing_scale ** 2) if cfg.processing_scale > 0 else 0
                
                # ĐỊNH NGHĨA KHI NÀO LÀ STRAIGHT (CHẾ ĐỘ FITLINE):
                # 1. Phải NHÌN THẤY đường ở xa (pixel_count > ngưỡng)
                # 2. Vạch kẻ đường phải SONG SONG với trục dọc (Góc lệch < Angle Thresh)
                
                angle_deg = 0.0
                line_pts = None
                
                pix_thresh = getattr(cfg, 'horizon_pix_thresh', 50)
                
                if pixel_count > pix_thresh:
                    contours, _ = cv2.findContours(h_mask_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Scale area threshold
                        min_area_proc = 10 * (cfg.processing_scale ** 2) if cfg.processing_scale > 0 else 10
                        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area_proc]
                        if valid_contours:
                            img_center_proc = h_mask_proc.shape[1] / 2.0
                            center_zone_proc = getattr(cfg, 'horizon_center_zone', 120) * cfg.processing_scale
                            
                            best_contour = None
                            min_dist = float('inf')
                            
                            for c in valid_contours:
                                M = cv2.moments(c)
                                if M["m00"] > 0:
                                    cx = int(M["m10"] / M["m00"])
                                    dist = abs(cx - img_center_proc)
                                    
                                    # Chỉ lấy line trong Center_Zone
                                    if dist > center_zone_proc:
                                        continue
                                        
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_contour = c
                            
                            if best_contour is not None:
                                import math
                                [vx, vy, x, y] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                                vx = float(vx[0])
                                vy = float(vy[0])
                                x = float(x[0])
                                y = float(y[0])
                                
                                # Đưa x, y về hệ quy chiếu ảnh gốc
                                if cfg.processing_scale > 0:
                                    x = x / cfg.processing_scale
                                    y = y / cfg.processing_scale
                                
                                if vy == 0:
                                    angle_deg = 90.0
                                else:
                                    angle_deg = math.degrees(math.atan(abs(vx / vy)))
                                    
                                # Tính tọa độ để vẽ đường
                                h_height = h_end - h_start
                                top_x = int(x + (0 - y) * (vx / vy)) if vy != 0 else int(x)
                                bot_x = int(x + (h_height - y) * (vx / vy)) if vy != 0 else int(x)
                                
                                top_x = max(-10000, min(10000, top_x))
                                bot_x = max(-10000, min(10000, bot_x))
                                line_pts = ((int(top_x), int(h_start)), (int(bot_x), int(h_end)))
                                
                                angle_thresh = getattr(cfg, 'horizon_angle_thresh', 15.0) 
                                
                                # Ngưỡng góc để quyết định (có Hysteresis nhẹ)
                                if self.prev_horizon_state == "STRAIGHT":
                                    effective_thresh = angle_thresh + 2.0
                                elif self.prev_horizon_state == "CURVE":
                                    effective_thresh = angle_thresh - 2.0
                                else:
                                    effective_thresh = angle_thresh
                                    
                                # Nếu góc chéo quá lớn -> CUA
                                if angle_deg > effective_thresh:
                                    horizon_state = "CURVE"
                                    box_color = (0, 0, 255) # Đỏ
                                else:
                                    # ĐƯỜNG THẲNG TẮP!
                                    horizon_state = "STRAIGHT"
                                    box_color = (0, 255, 0) # Xanh = Phóng
                            else:
                                horizon_state = "CURVE"
                                box_color = (0, 0, 255)
                        else:
                            horizon_state = "CURVE"
                            box_color = (0, 0, 255)
                    else:
                        horizon_state = "CURVE"
                        box_color = (0, 0, 255)
                else:
                    # Không thấy gì ở xa -> Đường đã uốn cong mất tiêu rồi
                    horizon_state = "CURVE"
                    box_color = (0, 0, 255) # Đỏ
                
                self.prev_horizon_state = horizon_state
                
                # Hiển thị Debug
                if self.visualizer is not None or self.video_writer is not None:
                    horizon_roi = frame[h_start:h_end, :]
                    tint = np.zeros_like(horizon_roi)
                    # Resize mask đã xử lý lên để vẽ đè lên frame gốc
                    h_mask_disp = cv2.resize(h_mask_proc, (horizon_roi.shape[1], horizon_roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                    tint[h_mask_disp == 255] = [255, 0, 255] # Màu Tím
                    frame[h_start:h_end, :] = cv2.addWeighted(horizon_roi, 0.6, tint, 0.4, 0)
                    
                    # Trục ảnh giữa và Center Zone
                    img_center = int(frame.shape[1] / 2.0)
                    center_zone = getattr(cfg, 'horizon_center_zone', 120)
                    cv2.line(frame, (img_center, h_start), (img_center, h_end), (0, 255, 255), 1)
                    
                    left_b = img_center - center_zone
                    right_b = img_center + center_zone
                    cv2.line(frame, (left_b, h_start), (left_b, h_end), (128, 128, 128), 1)
                    cv2.line(frame, (right_b, h_start), (right_b, h_end), (128, 128, 128), 1)
                    
                    # Vẽ đường nối các điểm
                    if line_pts is not None:
                        cv2.line(frame, line_pts[0], line_pts[1], box_color, 3)
                    
                    # Ghi thông số lên khung
                    cv2.putText(frame, f"Pix:{pixel_count} Angle:{angle_deg:.1f}", (frame.shape[1] - 180, h_start + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            else:
                box_color = (0, 255, 255) # Lỗi khung
                
            # Vẽ Box viền ngoài của Horizon Scanner và trạng thái
            cv2.rectangle(frame, (0, h_start), (frame.shape[1], h_end), box_color, 2)
            cv2.putText(frame, f"Horizon: {horizon_state}", (10, h_start - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            self.prev_horizon_state = horizon_state

        # ---- 4. BEV transform ----
        bev_mask = self.bev.warp_to_bev(mask)

        # ---- 5. Lane detection (Histogram + Sliding Window + RANSAC) ----
        detection = self.lane_detector.detect(bev_mask)

        # ---- 6. Geometry validation + center reconstruction ----
        prev_state = self.state_estimator.state
        observation = self.geometry.process(detection, prev_state)

        # ---- 7. Temporal state estimation ----
        lane_state = self.state_estimator.update(observation)

        # ---- 8. Trajectory generation ----
        traj = self.trajectory_gen.generate(lane_state, self.current_speed)

        # ---- 9. Lateral controller (identical to V3) ----
        if cfg.stanley_enabled and lane_state.tracking_state == TrackingState.TRACKING:
            steer_raw = self.stanley_controller.compute(
                lane_state.heading_error,
                lane_state.lateral_error_m,
                lane_state.curvature,
                self.current_speed
            )
        else:
            # Pure Pursuit (default)
            if traj.target is not None:
                steer_raw = self.pp_controller.compute(traj.target, traj.lookahead_m)
            else:
                steer_raw = 0.0

        # ---- 10. Steering filter (V3.1: speed-adaptive) ----
        # V3.1: Pass current throttle as speed proxy for adaptive gain
        steer_filtered = self.steering_filter.filter(
            steer_raw, current_speed=self.current_throttle)

        # ---- 10.5. Area Heuristic V2 (V3.1: deadband + speed-dependent) ----
        if lane_state.centerline_poly is not None and lane_state.tracking_state == TrackingState.TRACKING:
            y_vals = np.arange(0, pcfg.image_height)
            poly_x = np.polyval(lane_state.centerline_poly, y_vals)
            poly_x_clipped = np.clip(poly_x, 0, pcfg.image_width)

            area_left = np.sum(poly_x_clipped)
            total_area = pcfg.image_width * pcfg.image_height
            area_right = total_area - area_left

            if total_area > 0:
                area_ratio = (area_left - area_right) / total_area

                # V3.1: Deadband — ignore tiny ratios (noise on oval)
                if abs(area_ratio) < cfg.area_deadband:
                    area_ratio = 0.0

                # V3.1: Speed-dependent gain — less correction at high speed
                speed_factor = 1.0 - 0.5 * (self.current_throttle / cfg.max_speed) \
                    if cfg.max_speed > 0 else 1.0

                area_steer = cfg.area_k * area_ratio * speed_factor
                
                # Bù trừ hợp lý giữa Pure Pursuit và Area Heuristic để chống Oversteer:
                if steer_filtered * area_steer > 0:
                    # Cùng chiều: Dùng tiệm cận (asymptotic addition). 
                    # Nếu steer_filtered đã lớn, lực bồi thêm của area_steer sẽ bị giảm đi.
                    # VD: steer=0.8, area=0.2 -> bù thêm 0.2 * (1 - 0.8) = 0.04 -> tổng 0.84 (không bị giật)
                    steer_filtered += area_steer * (1.0 - min(1.0, abs(steer_filtered)))
                else:
                    steer_filtered += area_steer

            steer_filtered = max(-1.0, min(1.0, steer_filtered))
            
            # [V3.1 FIX] Cập nhật lại trạng thái bên trong của bộ lọc vô lăng 
            # để tránh bị kẹt (lệch pha) ở các frame tiếp theo
            self.steering_filter.prev_output = steer_filtered

        # ---- 11. Speed controller (V3.1: predictive curvature + early brake) ----
        throttle = self.speed_controller.compute(
            traj.max_upcoming_curvature,  # [V3.1] Use max curvature ahead, not just under vehicle
            lane_state.confidence,
            lane_state.tracking_state,
            actual_speed=None,  # TODO: encoder feedback
            horizon_state=horizon_state
        )

        # V3.1: Track throttle for speed-adaptive steering
        self.current_throttle = throttle

        # ---- 12. Debug visualization (V3.1: conditional) ----
        dashboard = None
        if self.visualizer is not None:
            # Need to upscale bev_mask for visualization if downscaled
            if self.use_downscale:
                bev_mask_viz = cv2.resize(bev_mask, (cfg.image_width, cfg.image_height))
            else:
                bev_mask_viz = bev_mask
            dashboard = self.visualizer.render(
                frame, bev_mask_viz, detection, lane_state, traj,
                steer_raw, steer_filtered, throttle,
                self.current_fps, raw_mask=mask
            )

        # ---- 13. Logging ----
        if self.logger:
            self.logger.log(lane_state, traj, steer_raw, steer_filtered, throttle)

        # FPS counter
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.fps_timer
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = now

        return steer_filtered, throttle, dashboard, lane_state

    # ==================================================================
    # RUN MODES
    # ==================================================================

    def run_ros(self):
        """Main loop for ROS mode (on JetRacer)."""
        print("=== SPEED RACING V3.1.00 (ROS Mode) ===")
        time.sleep(2)  # Wait for camera
        print("Starting...")

        rate = rospy.Rate(self.cfg.loop_rate)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            frame = self.latest_image.copy()
            steer, throttle, dashboard, lane_state = self.process_frame(frame)

            # Apply steering inversion if needed
            steer_out = -steer if self.cfg.steer_invert else steer

            # Send commands
            if lane_state.tracking_state == TrackingState.E_STOP:
                continue
            else:
                self.racer.steer(steer_out, throttle)

            # Record video (only if explicitly enabled and dashboard exists)
            if self.video_writer and dashboard is not None:
                self.video_writer.write(dashboard)

            # V3.1: Print FPS periodically in ROS mode
            if self.frame_count == 0 and self.current_fps > 0:
                rospy.loginfo_throttle(5,
                    f"V3.1 FPS={self.current_fps:.1f} thr={throttle:.3f} steer={steer:.3f}")

            rate.sleep()

        self._cleanup()

    def run_offline_video(self, video_path):
        """Offline replay mode — process a recorded video file."""
        print(f"=== SPEED RACING V3.1.00 (Offline Video: {video_path}) ===")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            return

        total_frames = 0
        start_time = time.time()
        
        cv2.namedWindow('V3.1 TUNER', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('V3.1 TUNER', 400, 700)
        
        # Thêm Live Calibrator (FULL)
        def nothing(x): pass
        cv2.createTrackbar('H_MIN', 'V3.1 TUNER', self.proc_cfg.hsv_h1_min, 179, nothing)
        cv2.createTrackbar('S_MIN', 'V3.1 TUNER', self.proc_cfg.hsv_s_min, 255, nothing)
        cv2.createTrackbar('V_MIN', 'V3.1 TUNER', self.proc_cfg.hsv_v_min, 255, nothing)
        
        cv2.createTrackbar('H_MAX', 'V3.1 TUNER', self.proc_cfg.hsv_h1_max, 179, nothing)
        cv2.createTrackbar('S_MAX', 'V3.1 TUNER', self.proc_cfg.hsv_s_max, 255, nothing)
        cv2.createTrackbar('V_MAX', 'V3.1 TUNER', self.proc_cfg.hsv_v_max, 255, nothing)
        
        cv2.createTrackbar('Blur_Size', 'V3.1 TUNER', self.proc_cfg.morph_kernel_size, 15, nothing)
        cv2.createTrackbar('ROI_Y(%)', 'V3.1 TUNER', int(self.proc_cfg.roi_y_start * 100), 100, nothing)
        cv2.createTrackbar('LAB_A_MIN', 'V3.1 TUNER', getattr(self.proc_cfg, 'lab_a_min', 0), 255, nothing)
        
        cv2.createTrackbar('Horizon_Y1', 'V3.1 TUNER', self.proc_cfg.horizon_scan_y_start, 480, nothing)
        cv2.createTrackbar('Horizon_Y2', 'V3.1 TUNER', self.proc_cfg.horizon_scan_y_end, 480, nothing)
        cv2.createTrackbar('Pix_Thresh', 'V3.1 TUNER', getattr(self.proc_cfg, 'horizon_pix_thresh', 50), 500, nothing)
        cv2.createTrackbar('Angle_Thresh', 'V3.1 TUNER', int(self.proc_cfg.horizon_angle_thresh), 90, nothing)
        cv2.createTrackbar('Center_Zone', 'V3.1 TUNER', self.proc_cfg.horizon_center_zone, 320, nothing)

        while True:
            # Cập nhật thông số từ Trackbar vào config (Live Update)
            try:
                # 1. HSV
                self.proc_cfg.hsv_h1_min = cv2.getTrackbarPos('H_MIN', 'V3.1 TUNER')
                self.proc_cfg.hsv_s_min = cv2.getTrackbarPos('S_MIN', 'V3.1 TUNER')
                self.proc_cfg.hsv_v_min = cv2.getTrackbarPos('V_MIN', 'V3.1 TUNER')
                self.proc_cfg.hsv_s_min_far = self.proc_cfg.hsv_s_min
                self.proc_cfg.hsv_v_min_far = self.proc_cfg.hsv_v_min
                
                self.proc_cfg.hsv_h1_max = cv2.getTrackbarPos('H_MAX', 'V3.1 TUNER')
                self.proc_cfg.hsv_s_max = cv2.getTrackbarPos('S_MAX', 'V3.1 TUNER')
                self.proc_cfg.hsv_v_max = cv2.getTrackbarPos('V_MAX', 'V3.1 TUNER')
                
                # 2. Blur / Filter
                blur = cv2.getTrackbarPos('Blur_Size', 'V3.1 TUNER')
                if blur % 2 == 0: blur += 1
                self.proc_cfg.morph_kernel_size = max(1, blur)
                self.proc_cfg.roi_y_start = cv2.getTrackbarPos('ROI_Y(%)', 'V3.1 TUNER') / 100.0
                
                lab = cv2.getTrackbarPos('LAB_A_MIN', 'V3.1 TUNER')
                self.proc_cfg.lab_a_min = lab
                self.proc_cfg.use_lab_constraint = (lab > 0)
                
                # 3. Horizon Scanner
                hy1 = cv2.getTrackbarPos('Horizon_Y1', 'V3.1 TUNER')
                hy2 = cv2.getTrackbarPos('Horizon_Y2', 'V3.1 TUNER')
                if hy1 > hy2: hy1, hy2 = hy2, hy1
                
                self.proc_cfg.horizon_scan_y_start = hy1
                self.proc_cfg.horizon_scan_y_end = hy2
                self.proc_cfg.horizon_pix_thresh = cv2.getTrackbarPos('Pix_Thresh', 'V3.1 TUNER')
                self.proc_cfg.horizon_angle_thresh = float(cv2.getTrackbarPos('Angle_Thresh', 'V3.1 TUNER'))
                self.proc_cfg.horizon_center_zone = cv2.getTrackbarPos('Center_Zone', 'V3.1 TUNER')
            except:
                pass
                
            ret, frame = cap.read()
            if not ret:
                break

            steer, throttle, dashboard, lane_state = self.process_frame(frame)

            total_frames += 1
            if total_frames % 20 == 0:
                elapsed = time.time() - start_time
                avg_fps = total_frames / elapsed if elapsed > 0 else 0
                print(f"Processed {total_frames} frames... "
                      f"Avg FPS: {avg_fps:.1f} (Current: {self.current_fps:.1f}) "
                      f"thr={throttle:.3f} steer={steer:.3f}")

            if dashboard is not None:
                cv2.imshow('V3.1 Dashboard', dashboard)

                if self.video_writer:
                    self.video_writer.write(dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Tự động lưu thông số vào config.py
                print("\n=== SAVING TUNED PARAMETERS TO CONFIG.PY ===")
                import re
                config_path = os.path.join(os.path.dirname(__file__), 'config.py')
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Danh sách các thông số cần update
                    params_to_update = {
                        'hsv_h1_min': self.proc_cfg.hsv_h1_min,
                        'hsv_s_min': self.proc_cfg.hsv_s_min,
                        'hsv_v_min': self.proc_cfg.hsv_v_min,
                        'hsv_h1_max': self.proc_cfg.hsv_h1_max,
                        'hsv_s_max': self.proc_cfg.hsv_s_max,
                        'hsv_v_max': self.proc_cfg.hsv_v_max,
                        'morph_kernel_size': self.proc_cfg.morph_kernel_size,
                        'roi_y_start': self.proc_cfg.roi_y_start,
                        'use_lab_constraint': self.proc_cfg.use_lab_constraint,
                        'lab_a_min': getattr(self.proc_cfg, 'lab_a_min', 0),
                        'horizon_scan_y_start': self.proc_cfg.horizon_scan_y_start,
                        'horizon_scan_y_end': self.proc_cfg.horizon_scan_y_end,
                        'horizon_pix_thresh': getattr(self.proc_cfg, 'horizon_pix_thresh', 50),
                        'horizon_angle_thresh': self.proc_cfg.horizon_angle_thresh,
                        'horizon_center_zone': self.proc_cfg.horizon_center_zone
                    }
                    
                    for var_name, val in params_to_update.items():
                        # Regex tìm kiếm khai báo biến: "tên_biến: kiểu = giá_trị" hoặc "tên_biến = giá_trị"
                        pattern = r"(^\s*" + var_name + r"\s*(?::\s*[a-zA-Z_]+)?\s*=\s*)([^#\n]+)"
                        
                        # Định dạng lại giá trị
                        if isinstance(val, bool):
                            str_val = str(val) + "   "
                        elif isinstance(val, float):
                            str_val = f"{val:.2f}   "
                        else:
                            str_val = f"{val}        "
                            
                        content = re.sub(pattern, r"\g<1>" + str_val, content, flags=re.MULTILINE)
                        print(f"Updated {var_name} = {val}")
                        
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
                    print("✅ LƯU THÀNH CÔNG VÀO config.py!")
                except Exception as e:
                    print(f"❌ Lỗi khi lưu config: {e}")
                    
                print("========================================================\n")
                break
            elif key == ord(' '):
                cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()
        self._cleanup()
        print("Offline replay complete.")

    def run_offline_webcam(self):
        """Offline mode with laptop webcam."""
        print("=== SPEED RACING V3.1.00 (Offline Webcam) ===")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            steer, throttle, dashboard, lane_state = self.process_frame(frame)

            if dashboard is not None:
                cv2.imshow('V3.1 Dashboard', dashboard)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._cleanup()

    def _cleanup(self):
        """Release all resources."""
        self.racer.stop()
        if self.logger:
            self.logger.close()
            print(f"Log saved: {self.logger.log_path}")
        if self.video_writer:
            self.video_writer.release()
            print("Video saved.")


# ==================================================================
# ENTRY POINT
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description='Speed Racing V3.1.00')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file for offline replay')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug visualizer (reduces FPS)')
    parser.add_argument('--full-res', action='store_true',
                        help='Process at full 640x480 (slower but more accurate)')
    parser.add_argument('--max-speed', type=float, default=None,
                        help='Override max_speed (e.g., 0.50, 0.55, 0.60)')
    args = parser.parse_args()

    config = make_v31_config()

    # Apply CLI overrides
    if args.debug:
        config.debug_mode = True
    if args.full_res:
        config.processing_scale = 1.0
    if args.max_speed is not None:
        config.max_speed = args.max_speed
        config.cruise_speed = min(config.cruise_speed, args.max_speed)
        print(f"Max speed override: {args.max_speed}")

    # For offline replay, always enable debug
    if args.video:
        config.debug_mode = True

    app = SpeedRacingV31(config=config, video_path=args.video)

    if args.video:
        app.run_offline_video(args.video)
    elif HAS_ROS:
        try:
            app.run_ros()
        except rospy.ROSInterruptException:
            pass
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            app.racer.stop()
    else:
        app.run_offline_webcam()


if __name__ == '__main__':
    main()
