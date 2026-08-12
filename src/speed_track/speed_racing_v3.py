#!/usr/bin/env python3
"""
Speed Track Racing V3.0 — Multi-Lane Temporal Tracking Architecture

Complete rewrite of the lane following pipeline:
  Camera → Undistort → ROI → HSV → Morph → BEV
  → Histogram + Sliding Window → RANSAC per line
  → Geometry Validation → Center Reconstruction
  → Temporal EMA Filter → Look-ahead Trajectory
  → Pure Pursuit / Stanley → Steering Filter → Speed Controller

Preserves all existing hardware/ROS interfaces:
  - ROS topic: /csi_cam_0/image_raw (sensor_msgs/Image)
  - ROS topic: /scan (sensor_msgs/LaserScan) — reserved for future obstacle avoidance
  - Hardware: RacerController (NvidiaRacecar / JetBot fallback / Mock)
  - Logging: CSV + AVI video

Run modes:
  - ROS mode (on JetRacer): `rosrun speed_track speed_racing_v3.py`
  - Offline mode (laptop):    `python speed_racing_v3.py`
  - Offline video replay:     `python speed_racing_v3.py --video path/to/video.avi`
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
        self.pp_controller = PurePursuitController(self.cfg)
        self.stanley_controller = StanleyController(self.cfg)
        self.steering_filter = SteeringFilter(self.cfg)
        self.speed_controller = SpeedController(self.cfg)
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
        self.frame_count = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

    # ==================================================================
    # ROS CALLBACKS
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
    # CORE PIPELINE — Process one frame
    # ==================================================================

    def process_frame(self, frame):
        """Run the complete V3 pipeline on a single frame.

        Args:
            frame: BGR image (640x480).

        Returns:
            (steer_filtered, throttle, dashboard_image, lane_state)
        """
        cfg = self.cfg

        # Ensure correct size
        if frame.shape[:2] != (cfg.image_height, cfg.image_width):
            frame = cv2.resize(frame, (cfg.image_width, cfg.image_height))

        # ---- 1. Undistortion ----
        frame_undist = self.undistorter.process(frame)

        # ---- 2. ROI crop (for segmentation only, BEV uses full frame) ----
        roi_y_start = int(cfg.roi_y_start * cfg.image_height)
        roi_frame = frame_undist.copy()
        # Zero out above ROI to avoid detecting sky/far noise
        roi_frame[:roi_y_start, :] = 0

        # ---- 3. Color segmentation ----
        mask = self.segmenter.process(roi_frame)

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

        # ---- 9. Lateral controller ----
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

        # ---- 10. Steering filter ----
        steer_filtered = self.steering_filter.filter(steer_raw)

        # ---- 10.5. Area Heuristic (Ý TƯỞNG 2: Mạng lưới an toàn diện tích) ----
        if lane_state.centerline_poly is not None:
            # Tìm vị trí vạch vàng ở đáy BEV
            bottom_y = self.cfg.image_height
            center_x_bottom = int(np.polyval(lane_state.centerline_poly, bottom_y))
            center_x_bottom = max(0, min(self.cfg.image_width, center_x_bottom))
            
            # Cắt đôi bức ảnh bằng vạch vàng và đếm điểm ảnh trắng
            area_left = np.count_nonzero(bev_mask[:, :center_x_bottom])
            area_right = np.count_nonzero(bev_mask[:, center_x_bottom:])
            
            # Heuristic: Nếu một bên có quá nhiều vạch (diện tích > 175%), ép vô lăng bẻ về hướng ngược lại
            # Lưu ý: Do steering của xe có thể đảo chiều (steer_invert), ta điều chỉnh steer_filtered.
            # Với convention của V3, bẻ trái là âm (-), bẻ phải là dương (+)
            if area_right > area_left * 1.75:
                steer_filtered -= 0.3  # Ép bẻ Trái
            elif area_left > area_right * 1.75:
                steer_filtered += 0.3  # Ép bẻ Phải
                
            # Đảm bảo steering không vượt ngưỡng [-1.0, 1.0]
            steer_filtered = max(-1.0, min(1.0, steer_filtered))

        # ---- 11. Speed controller ----
        throttle = self.speed_controller.compute(
            traj.curvature,
            lane_state.confidence,
            lane_state.tracking_state,
            actual_speed=None  # TODO: encoder feedback
        )

        # ---- 12. Debug visualization ----
        dashboard = self.visualizer.render(
            frame, bev_mask, detection, lane_state, traj,
            steer_raw, steer_filtered, throttle,
            self.current_fps
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
        print("=== SPEED RACING V3 (ROS Mode) ===")
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
                # self.racer.stop()
                # rospy.logwarn("E_STOP — all motors stopped.")
                # break
                continue
            else:
                self.racer.steer(steer_out, throttle)

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
