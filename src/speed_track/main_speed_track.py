#!/usr/bin/env python3
"""
Speed Track — Main Entry Point (V3 Architecture)

Pipeline:
  Camera → Undistort → ROI → HSV → Morph → BEV
  → Histogram + Sliding Window → RANSAC per line
  → Geometry Validation → Center Reconstruction
  → Temporal EMA Filter → Look-ahead Trajectory
  → Pure Pursuit / Stanley → Steering Filter
  → Obstacle Avoidance (APF + State Machine) → Speed Controller

Hardware/ROS interfaces:
  - ROS topic: /csi_cam_0/image_raw (sensor_msgs/Image)
  - ROS topic: /scan (sensor_msgs/LaserScan) — obstacle avoidance
  - Hardware: RacerController (NvidiaRacecar / JetBot fallback / Mock)
  - Logging: CSV + AVI video

Run modes:
  - ROS mode (on JetRacer): `python3 src/speed_track/main_speed_track.py`
  - Offline video replay:    `python3 src/speed_track/main_speed_track.py --video path/to/video.avi`
"""

import sys
sys.path.append("../../")

import os
import cv2
import math
import time
import threading
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

# V3 modules
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
from src.speed_track.control.obstacle_avoidance import ObstacleAvoidance, AvoidState
from src.speed_track.debug.visualizer import DebugVisualizer
from src.speed_track.debug.logger import V3Logger


class SpeedRacingV3:
    """Main V3 runner — integrates all modules into a real-time pipeline."""

    def __init__(self, config=None, video_path=None):
        """
        Args:
            config: V3Config instance. Uses defaults if None.
            video_path: Path to a video file for offline replay. None for live camera.
        """
        self.cfg = config or V3Config()
        self.video_path = video_path

        # ---- Initialize pipeline modules ----
        self.undistorter = Undistorter(self.cfg)
        self.segmenter = ColorSegmenter(self.cfg)
        self.bev = BEVTransform(self.cfg)
        self.lane_detector = MultiLaneDetector(self.cfg)
        self.geometry = LaneGeometry(self.cfg, self.bev)
        self.state_estimator = LaneStateEstimator(self.cfg, self.bev)
        self.trajectory_gen = TrajectoryGenerator(self.cfg, self.bev)
        from src.speed_track.control.corridor_planner import CorridorPlanner
        self.corridor_planner = CorridorPlanner(self.cfg, self.bev, self.trajectory_gen)
        self.pp_controller = PurePursuitController(self.cfg)
        self.stanley_controller = StanleyController(self.cfg)
        self.steering_filter = SteeringFilter(self.cfg)
        self.speed_controller = SpeedController(self.cfg)
        self.obstacle_avoidance = ObstacleAvoidance(self.cfg)
        self.visualizer = DebugVisualizer(self.cfg)

        # ---- Hardware ----
        self.racer = RacerController()
        self.racer.stop()

        # ---- ROS ----
        self.latest_image = None
        self.latest_scan = None

        if HAS_ROS and video_path is None:
            rospy.init_node(self.cfg.node_name, anonymous=True)
            rospy.Subscriber(self.cfg.camera_topic, Image, self._cam_cb)
            rospy.Subscriber(self.cfg.lidar_topic, LaserScan, self._lidar_cb)

        # ---- Logging ----
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        self.logger = V3Logger(log_dir, prefix='v3') if self.cfg.log_csv else None
        self.video_writer = None

        if self.cfg.record_video:
            ts = time.strftime('%Y%m%d_%H%M%S')
            vid_path = os.path.join(log_dir, f'v3_{ts}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(
                vid_path, fourcc, self.cfg.video_fps,
                (self.cfg.image_width * 2, self.cfg.image_height)
            )
            print(f"Recording video to: {vid_path}")

        # ---- State ----
        self.current_speed = 0.0  # Estimated or measured speed (m/s)
        self.last_steer = 0.0     # Last normalized steering [-1.0, 1.0]
        self.last_steer_rad = 0.0 # Last physical steering angle in radians (+right, -left)
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

        # ---- Startup Calibration Validation Guard ----
        self._startup_widths = []
        self._startup_frame_count = 0
        self.calibration_valid = True
        self.calibration_checked = False
        self.latest_image_received_time = 0.0
        self._sensor_lock = threading.Lock()

    # ==================================================================
    # ROS CALLBACKS & THREAD-SAFE SENSOR SNAPSHOT
    # ==================================================================

    def _cam_cb(self, msg):
        """ROS camera callback — converts Image msg to BGR numpy array with thread-safe lock."""
        try:
            now = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp.to_sec() if hasattr(msg.header.stamp, 'to_sec') else float(msg.header.stamp)
            else:
                stamp = now

            if 'compressed' in msg.encoding:
                img = cv2.imdecode(
                    np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                raw_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1)
                if 'rgb' in msg.encoding:
                    img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                else:
                    img = raw_img

            with self._sensor_lock:
                self.latest_image = img
                self.latest_image_stamp = stamp
                self.latest_image_received_time = now
        except Exception as e:
            if HAS_ROS:
                rospy.logerr_throttle(5, f"Camera callback error: {e}")

    def _lidar_cb(self, msg):
        """ROS LiDAR callback — stores latest scan with thread-safe lock."""
        with self._sensor_lock:
            self.latest_scan = msg

    def get_sensor_snapshot(self):
        """Get an atomic thread-safe snapshot of camera frame, timestamp, and LiDAR scan."""
        with self._sensor_lock:
            img = self.latest_image.copy() if self.latest_image is not None else None
            stamp = self.latest_image_stamp
            recv_time = self.latest_image_received_time
            scan = self.latest_scan
        return img, stamp, recv_time, scan

    # ==================================================================
    # CORE PIPELINE — Process one frame
    # ==================================================================

    def to_hardware_steering(self, delta_rad: float) -> float:
        """Convert physical steering angle (rad, +right/-left) to normalized hardware steering [-1.0, 1.0].
        Hardware inversion (steer_invert) is applied ONLY at this actuator stage.
        """
        normalized = delta_rad / self.cfg.max_steer_rad
        clamped = max(-1.0, min(1.0, normalized))
        return -clamped if self.cfg.steer_invert else clamped

    def process_frame(self, frame, timestamp=None):
        """Process a single camera frame through the full perception-estimation-control pipeline.

        Args:
            frame: BGR image from camera (numpy array).
            timestamp: Camera frame timestamp in seconds (optional).

        Returns:
            tuple: (steer_hw, throttle, dashboard_image, lane_state)
        """
        cfg = self.cfg
        t_cam = timestamp if timestamp is not None else time.time()
        now_t = time.time()
        dt = max(0.001, min(0.10, now_t - self.last_frame_time)) if hasattr(self, 'last_frame_time') and self.last_frame_time else (1.0 / cfg.loop_rate)
        self.last_frame_time = now_t

        # ---- 1. Undistortion ----
        frame_undist = self.undistorter.process(frame)

        # ---- 2. ROI crop (for segmentation only, BEV uses full frame) ----
        roi_y_start = int(cfg.roi_y_start * cfg.image_height)
        roi_frame = frame_undist.copy()
        # Zero out above ROI to avoid detecting sky/far noise
        roi_frame[:roi_y_start, :] = 0

        # ---- 3. Color segmentation (HSV) ----
        mask = self.segmenter.process(roi_frame)

        # ---- 4. BEV transform ----
        bev_mask = self.bev.warp_to_bev(mask)

        # ---- 5. Lane detection (Histogram + Sliding Window + RANSAC + 3-Layer Fusion) ----
        detection = self.lane_detector.detect(bev_mask, current_steer=self.last_steer)

        # ---- 6. Geometry validation + center reconstruction ----
        prev_state = self.state_estimator.state
        observation = self.geometry.process(detection, prev_state)

        # Startup calibration guard: check measured width over initial dual-line frames
        if not self.calibration_checked and self.calibration_valid:
            if observation.valid and observation.method in ('L+R_midpoint', 'L+C+R_fused'):
                self._startup_widths.append(observation.lane_width_m)
                if len(self._startup_widths) >= cfg.startup_check_frames:
                    median_w = float(np.median(self._startup_widths))
                    expected_w = cfg.expected_lane_width_m
                    diff_pct = abs(median_w - expected_w) / expected_w
                    if diff_pct > cfg.startup_width_tolerance:
                        self.calibration_valid = False
                        print(f"\n[FATAL] BEV Calibration Invalid! Measured {median_w:.3f}m vs expected {expected_w:.3f}m (error {diff_pct*100:.1f}% > {cfg.startup_width_tolerance*100:.1f}%). Stopping.")
                    else:
                        print(f"\n[OK] Startup BEV Calibration Verified: Median Width = {median_w:.3f}m (error {diff_pct*100:.1f}%)\n")
                    self.calibration_checked = True
            else:
                self._startup_frame_count += 1
                timeout_frames = getattr(cfg, 'startup_timeout_frames', 60)
                if self._startup_frame_count >= timeout_frames:
                    self.calibration_valid = False
                    print(f"\n[FATAL] Startup Calibration Timeout! Dual boundaries not detected within {timeout_frames} frames. Stopping.\n")
                    self.calibration_checked = True

        if not self.calibration_valid:
            # Refuse to run on invalid BEV calibration or startup timeout
            self.racer.stop()
            return 0.0, 0.0, None, prev_state

        # ---- 7. Temporal state estimation ----
        n_lines = sum(1 for line in [detection.left, detection.center, detection.right] if line.detected)
        lane_state = self.state_estimator.update(
            observation,
            obstacle_near=False,
            n_lines_detected=n_lines,
        )

        # ---- 7.5. Corridor-Based Local Trajectory Planning & Safety Gating ----
        plan_res = self.corridor_planner.plan(
            lane_state, self.latest_scan, current_speed=self.current_speed,
            camera_timestamp=t_cam, current_steer_rad=self.last_steer_rad
        )

        if not plan_res.safe_to_proceed or plan_res.selected_trajectory is None:
            # Safe Emergency Stop: corridor blocked, timeout, or no collision-free path
            self.racer.stop()
            self.current_speed = 0.0
            return 0.0, 0.0, None, lane_state

        traj = plan_res.selected_trajectory

        # ---- 8. Steering: Single Source of Truth from Planner ----
        steer_cmd_rad = plan_res.selected_steer_rad
        steer_filtered_rad = self.steering_filter.filter_rad(steer_cmd_rad, dt=dt)
        self.last_steer_rad = steer_filtered_rad
        self.last_steer = steer_filtered_rad / cfg.max_steer_rad

        # Normalized hardware steering (with inversion ONLY applied here)
        steer_hw = self.to_hardware_steering(steer_filtered_rad)

        # ---- 9. Speed controller ----
        throttle = self.speed_controller.compute(
            traj.curvature,
            lane_state.confidence,
            lane_state.tracking_state,
            actual_speed=None,  # TODO: encoder feedback
            reconstruction_method=lane_state.reconstruction_method
        )

        # Scale throttle by planner speed factor
        throttle *= plan_res.speed_factor

        # Update dynamic open-loop speed estimation (P0 Fix)
        target_v = throttle / max(0.1, cfg.speed_to_throttle_factor)
        target_v = min(cfg.max_speed, max(0.0, target_v))
        self.current_speed = 0.7 * self.current_speed + 0.3 * target_v

        # ---- 10. Debug visualization ----
        dashboard = self.visualizer.render(
            frame, bev_mask, detection, lane_state, traj,
            steer_cmd_rad / cfg.max_steer_rad, self.last_steer, throttle,
            self.current_fps
        )

        # ---- 11. Logging ----
        if self.logger:
            self.logger.log(lane_state, traj, steer_cmd_rad / cfg.max_steer_rad, self.last_steer, throttle)

        # FPS counter
        self.frame_count += 1
        elapsed = now_t - self.fps_timer
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = now_t

        return steer_hw, throttle, dashboard, lane_state

    # ==================================================================
    # RUN MODES
    # ==================================================================

    def run_ros(self):
        """Main loop for ROS mode (on JetRacer)."""
        print("=== SPEED RACING V3 (ROS Mode) ===")
        time.sleep(2)  # Wait for camera
        print("Starting...")

        rate = rospy.Rate(self.cfg.loop_rate)

        while not rospy.is_shutdown():
            now = time.time()
            frame, t_cam, t_recv, scan = self.get_sensor_snapshot()

            if frame is None:
                rate.sleep()
                continue

            # Camera Watchdog: Check if camera stream has stalled or frozen
            cam_timeout = getattr(self.cfg, 'camera_timeout_s', 0.25)
            if t_recv > 0.0 and (now - t_recv) > cam_timeout:
                self.racer.stop()
                if HAS_ROS:
                    rospy.logwarn_throttle(2, f"Camera Watchdog Timeout: No new frame for {now - t_recv:.3f}s. Motors stopped.")
                rate.sleep()
                continue

            steer_hw, throttle, dashboard, lane_state = self.process_frame(frame, timestamp=t_cam)

            # Send commands
            if lane_state.tracking_state == TrackingState.E_STOP or not self.calibration_valid:
                self.racer.stop()
                rospy.logwarn("E_STOP or Invalid Calibration — all motors stopped.")
                if lane_state.tracking_state == TrackingState.E_STOP:
                    break
            elif throttle <= 0.0:
                self.racer.stop()
            else:
                self.racer.steer(steer_hw, throttle)

            # Record video
            if self.video_writer and dashboard is not None:
                self.video_writer.write(dashboard)

            rate.sleep()

        self._cleanup()

    def run_offline_video(self, video_path):
        """Offline replay mode — process a recorded video file."""
        print(f"=== SPEED RACING V3 (Offline Video: {video_path}) ===")
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
                print(f"Processed {total_frames} frames... Avg FPS: {avg_fps:.1f} (Current: {self.current_fps:.1f})")

            if dashboard is not None:
                # Show at manageable size
                display = cv2.resize(dashboard, (1280, 480))
                cv2.imshow('V3 Dashboard', display)

                if self.video_writer:
                    self.video_writer.write(dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Pause
                cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()
        self._cleanup()
        print("Offline replay complete.")

    def run_offline_webcam(self):
        """Offline mode with laptop webcam."""
        print("=== SPEED RACING V3 (Offline Webcam) ===")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            steer, throttle, dashboard, lane_state = self.process_frame(frame)

            if dashboard is not None:
                cv2.imshow('V3 Dashboard', dashboard)

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
    parser = argparse.ArgumentParser(description='Speed Racing V3')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file for offline replay')
    args = parser.parse_args()

    config = V3Config()
    app = SpeedRacingV3(config=config, video_path=args.video)

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
