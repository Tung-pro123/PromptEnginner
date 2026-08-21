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
    """Create a V3Config with V3.1 performance preset applied.

    All V3 defaults are preserved except where explicitly overridden.
    Each override is documented with rationale.
    """
    cfg = V3Config()

    # ---- FPS Optimization ----
    cfg.processing_scale = 1.0          # Trả lại phân giải gốc như V3 (640x480) để nhận diện chính xác
    cfg.debug_mode = False              # Skip visualizer rendering in ROS mode
    cfg.record_video = False            # No video recording (saves I/O)
    cfg.loop_rate = 30                  # Allow up to 30 Hz control loop

    # ---- Fix HSV Color Filter (Reject Background Noise) ----
    # Đã xóa các thông số HSV cứng (0, 59, 92) ở đây để V3.1 dùng chung 
    # bộ thông số lọc nhiễu gốc cực kỳ sạch của V3 trong config.py!
    
    # BẬT Lọc LAB để diệt sạch rác trắng/xám ở vùng Horizon mà không cần kéo HSV quá chặt
    cfg.use_lab_constraint = True
    cfg.lab_a_min = 132

    # ---- RANSAC/Window optimization ----
    cfg.sw_n_windows = 9                # 12 → 9 (sufficient for 240px height)
    cfg.ransac_max_trials = 15          # 30 → 15 (save 50% RANSAC time, because we now sample only 3 points)

    # ---- Speed boost (Predictive logic allows higher speeds) ----
    # Đã xoá ghi đè cứng. Mọi thông số tốc độ được sử dụng trực tiếp từ config.py.

    # ---- Speed-adaptive steering ----
    # Đã xoá ghi đè cứng. Mọi thông số steering được sử dụng trực tiếp từ config.py.

    # ---- Curvature-history speed (Disabled for multi-curve tracks) ----
    # Đã xoá ghi đè cứng. Mọi thông số sử dụng trực tiếp từ config.py.

    # ---- Predictive Braking & Horizon Scanner ----
    # Bật lại Horizon Scanner để phanh sớm cuối đường thẳng.
    cfg.horizon_warning_enabled = True
    cfg.horizon_scan_y_start = 215      # Kéo gần lại một chút (200 -> 215) để giữ được màu sắc
    cfg.horizon_scan_y_end = 260
    cfg.horizon_shift_thresh = 0.15

    # ---- Area heuristic v2 ----
    cfg.area_k = 0.15                   # Khôi phục sức mạnh như bản V3
    cfg.area_deadband = 0.0             # Tắt deadband như V3

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

        # ---- 3.5. Horizon Scanner (Dự báo Đường Thẳng / Khúc Cua) ----
        # Giải quyết Lỗi số 4 (Mù màu): Mặc định là UNKNOWN. Chỉ khi NHÌN RÕ vạch và KHÔNG LỆCH thì mới dám báo STRAIGHT.
        horizon_state = "UNKNOWN"
        if getattr(cfg, 'horizon_warning_enabled', False):
            # Scan the horizon band for lane shifts before they enter BEV
            h_start = int(getattr(cfg, 'horizon_scan_y_start', 215) * pcfg.processing_scale)
            h_end = int(getattr(cfg, 'horizon_scan_y_end', 260) * pcfg.processing_scale)
            
            # Ensure indices are valid
            h_start = max(0, min(h_start, mask.shape[0]))
            h_end = max(h_start, min(h_end, mask.shape[0]))
            
            horizon_mask = mask[h_start:h_end, :]
            
            # [Hiển thị Debug] Tô màu Tím (Magenta) cho các pixel lọt qua bộ lọc trong vùng Horizon
            if self.visualizer is not None or self.video_writer is not None:
                tint = np.zeros_like(frame[h_start:h_end, :])
                tint[horizon_mask == 255] = [255, 0, 255] # BGR: Magenta
                frame[h_start:h_end, :] = cv2.addWeighted(frame[h_start:h_end, :], 0.6, tint, 0.4, 0)
            
            # Find centroid of white pixels (lane markings) in the horizon
            M = cv2.moments(horizon_mask)
            
            # Vẽ Box của Horizon Scanner lên raw frame để Debug
            box_color = (0, 255, 0) # Green = an toàn
            
            if M["m00"] > 50:  # If enough lane pixels are visible
                cx = int(M["m10"] / M["m00"])
                img_center = mask.shape[1] / 2.0
                shift_ratio = abs(cx - img_center) / mask.shape[1]
                
                # If centroid is shifted > thresh off center, it's a sharp curve ahead
                if shift_ratio > getattr(cfg, 'horizon_shift_thresh', 0.15):
                    box_color = (0, 0, 255) # Red = phanh gấp
                    if cx < img_center:
                        horizon_state = "CURVE_LEFT"
                    else:
                        horizon_state = "CURVE_RIGHT"
                else:
                    # Đã nhìn thấy vạch và vạch thẳng! Đủ điều kiện an toàn để tăng tốc!
                    horizon_state = "STRAIGHT"
                    
                # Vẽ mũi tên chỉ hướng rẽ (từ giữa màn hình chĩa về hướng vạch kẻ đường)
                center_y = h_start + (h_end - h_start) // 2
                start_pt = (int(img_center), center_y)
                end_pt = (cx, center_y)
                
                # Chỉ vẽ mũi tên nếu vạch có xu hướng rẽ rõ ràng, nếu đi thẳng thì vẽ chấm
                if horizon_state != "STRAIGHT":
                    cv2.arrowedLine(frame, start_pt, end_pt, box_color, 3, tipLength=0.3)
                else:
                    cv2.circle(frame, (cx, center_y), 5, box_color, -1)
            else:
                # Không nhìn thấy vạch -> Đổi màu Box sang Vàng cảnh báo
                box_color = (0, 255, 255) # Yellow = mù/không chắc chắn
                
            # Vẽ Box viền ngoài của Horizon Scanner và trạng thái
            cv2.rectangle(frame, (0, h_start), (frame.shape[1], h_end), box_color, 2)
            cv2.putText(frame, f"Horizon: {horizon_state}", (10, h_start - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

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
        if lane_state.centerline_poly is not None:
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
                speed_factor = 1.0 - cfg.area_high_speed_penalty * (self.current_throttle / cfg.max_speed) \
                    if cfg.max_speed > 0 else 1.0

                area_steer = cfg.area_k * area_ratio * speed_factor
                
                # Bù trừ hợp lý giữa Pure Pursuit và Area Heuristic để chống Oversteer:
                if steer_filtered * area_steer > 0:
                    # Cùng chiều: Dùng tiệm cận (asymptotic addition). 
                    # Nếu steer_filtered đã lớn, lực bồi thêm của area_steer sẽ bị giảm đi.
                    # VD: steer=0.8, area=0.2 -> bù thêm 0.2 * (1 - 0.8) = 0.04 -> tổng 0.84 (không bị giật)
                    steer_filtered += area_steer * (1.0 - min(1.0, abs(steer_filtered)))
                else:
                    # Ngược chiều: Pure Pursuit đang bẻ lái sai so với mạng lưới an toàn (diện tích).
                    # Cho phép trừ trực tiếp để kéo vô lăng lại.
                    steer_filtered += area_steer

            steer_filtered = max(-1.0, min(1.0, steer_filtered))

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

        while True:
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
