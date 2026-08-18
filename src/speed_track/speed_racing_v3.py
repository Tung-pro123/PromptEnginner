#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed Track Racing V3.0 — Multi-Lane Temporal Tracking + LiDAR Cube (5cm-10cm) Obstacle Avoidance.

Pipeline Bám làn V3 Đỉnh cao:
  Camera -> Undistort -> ROI -> HSV -> Morph -> BEV
  -> Histogram + Sliding Window -> RANSAC per line
  -> Geometry Validation -> Center Reconstruction
  -> Temporal EMA Filter -> Look-ahead Trajectory
  -> Pure Pursuit / Stanley -> Steering Filter -> Speed Controller

Khử / Né vật cản LiDAR (Khối 5cm - 10cm):
  -> Front Wedge Scan (-35 deg -> +35 deg) + Percentile Noise Filter (>= 3 pts)
  -> Adaptive Dodge Offset (55px cho khối 5-7cm, 75px cho khối 10cm)
  -> Safety Steering Override (>= 0.28 khi né khối 10cm)
  -> Two-Stage Re-entering (Trả làn 2 giai đoạn sau khi qua hông 35cm)
  -> Web Live Stream (Port 8080) + CSV Telemetry + AVI Video Logger

Chạy trên xe:
    python3 src/speed_track/speed_racing_v3.py
"""

import sys
import os
import cv2
import math
import time
import argparse
import numpy as np
import csv

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

# 1. Bắt buộc loại bỏ python2.7 trước để Python 3 import cv2 & numpy chuẩn
sys.path = [p for p in sys.path if 'python2.7' not in p]

# 2. Nạp đường dẫn rospy cho Python 3
ros_paths = [
    "/opt/ros/melodic/lib/python2.7/dist-packages",
    "/media/jetson/ff2880cc-1a99-40bd-88c1-5cdc86fe9eed1/opt/ros/melodic/lib/python2.7/dist-packages"
]
for p in ros_paths:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

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

# V3 submodules (Nạp từ src/speed_track/)
try:
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
except ImportError:
    V3Config = None

from src.debug.web_viewer import start_web_stream_server, set_web_frame

class FSMObstacleState:
    NORMAL = "NORMAL"
    DODGING = "DODGING"
    REENTERING = "REENTERING"

class SpeedRacingV3:
    """Main V3 runner — integrates Multi-Lane Tracking with 5cm-10cm Cube Obstacle Avoidance."""

    def __init__(self, config=None, video_path=None):
        self.cfg = config or (V3Config() if V3Config else None)
        self.video_path = video_path

        # ---- Pipeline modules ----
        if V3Config and self.cfg:
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
            node_name = self.cfg.node_name if self.cfg else 'speed_racing_v3'
            cam_topic = self.cfg.camera_topic if self.cfg else '/csi_cam_0/image_raw'
            lidar_topic = self.cfg.lidar_topic if self.cfg else '/scan'
            
            rospy.init_node(node_name, anonymous=True)
            rospy.Subscriber(cam_topic, Image, self._cam_cb, queue_size=1)
            rospy.Subscriber(lidar_topic, LaserScan, self._lidar_cb, queue_size=1)

        # ---- FSM NÉ KHỐI LẬP PHƯƠNG (5cm - 10cm) ----
        self.obs_state = FSMObstacleState.NORMAL
        self.LIDAR_OFFSET_DEG = 180.0
        self.TRIGGER_DIST = 0.85         # Cự ly kích hoạt né 85cm (Tăng cự ly phát hiện sớm)
        self.SIDE_CLEAR_DIST = 0.35      # Khoảng cách an toàn hông xe 35cm
        self.DODGE_OFFSET_PX = 85.0      # Offset lách khối 10cm (Tăng biên độ lách rộng hơn)
        self.DODGE_OFFSET_SMALL_PX = 60.0# Offset lách khối 5-7cm
        self.RAMP_STEP_DODGE = 12.0      # Pixel/frame lách nhanh nhạy
        self.RAMP_STEP_RETURN = 4.0      # Pixel/frame trả mượt từ từ
        self.MIN_DODGE_DURATION = 1.2    # Giây. Khóa né tối thiểu (Có thể tinh chỉnh 1.0s - 1.5s)

        self.current_offset_px = 0.0
        self.target_offset_px = 0.0
        self.dodge_dir = 0.0
        self.dodge_start_time = 0.0
        self.reenter_start_time = 0.0
        self.clear_side_count = 0

        # ---- Dedicated CSV Telemetry & Video Logger ----
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')

        self.telemetry_path = os.path.join(log_dir, f'v3_obstacle_telemetry_{ts}.csv')
        self._telemetry_file = open(self.telemetry_path, 'w', newline='')
        self._telemetry_csv = csv.writer(self._telemetry_file)
        self._telemetry_csv.writerow([
            'timestamp_sec', 'obs_state', 'time_in_dodge_sec', 'front_dist_m', 'side_dist_m',
            'dodge_dir', 'current_offset_px', 'steer_raw', 'steer_filtered', 'throttle'
        ])

        self.logger = V3Logger(log_dir, prefix='v3') if (self.cfg and self.cfg.log_csv) else None
        self.video_writer = None

        if self.cfg and self.cfg.record_video:
            vid_path = os.path.join(log_dir, f'v3_obstacle_{ts}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(
                vid_path, fourcc, self.cfg.video_fps,
                (self.cfg.image_width * 2, self.cfg.image_height)
            )
            print(f"Recording video to: {vid_path}")

        # Web Stream Server (Port 8080)
        try:
            start_web_stream_server(port=8080)
        except Exception:
            pass

        self.current_speed = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                self.latest_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if 'rgb' in msg.encoding:
                    self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    self.latest_image = img
        except Exception as e:
            if HAS_ROS:
                rospy.logerr_throttle(5, f"Camera callback error: {e}")

    def _lidar_cb(self, msg):
        self.latest_scan = msg

    # ==================================================================
    # XỬ LÝ LIDAR PHÁT HIỆN & NÉ KHỐI LẬP PHƯƠNG (5CM - 10CM)
    # ==================================================================
    def process_lidar_cubes(self):
        if self.latest_scan is None:
            return float('inf'), None, 0

        msg = self.latest_scan
        front_dists = []
        front_angles = []

        for i, d in enumerate(msg.ranges):
            if not (msg.range_min < d < msg.range_max):
                continue
            deg = math.degrees(msg.angle_min + i * msg.angle_increment) + self.LIDAR_OFFSET_DEG
            angle = (deg + 180) % 360 - 180

            if -35 <= angle <= 35 and d < 1.2:
                front_dists.append(d)
                front_angles.append(angle)

        if not front_dists:
            return float('inf'), None, 0

        min_idx = np.argmin(front_dists)
        return front_dists[min_idx], front_angles[min_idx], len(front_dists)

    def check_side_clearance(self):
        if self.latest_scan is None:
            return True

        msg = self.latest_scan
        side_dists = []

        a_min = -150.0 if self.dodge_dir == 1.0 else -30.0
        a_max = 30.0 if self.dodge_dir == 1.0 else 150.0

        for i, d in enumerate(msg.ranges):
            if not (msg.range_min < d < msg.range_max):
                continue
            deg = math.degrees(msg.angle_min + i * msg.angle_increment) + self.LIDAR_OFFSET_DEG
            angle = (deg + 180) % 360 - 180

            if a_min <= angle <= a_max:
                side_dists.append(d)

        if not side_dists:
            return True

        return min(side_dists) >= self.SIDE_CLEAR_DIST

    def update_obstacle_fsm(self):
        min_dist, closest_angle, pt_count = self.process_lidar_cubes()
        now = time.time()

        # 1. NORMAL -> DODGING
        if self.obs_state == FSMObstacleState.NORMAL:
            self.target_offset_px = 0.0
            if min_dist < self.TRIGGER_DIST and pt_count >= 3:
                self.obs_state = FSMObstacleState.DODGING
                self.dodge_start_time = now
                self.clear_side_count = 0

                # Khối 10cm (to nhất, pt_count >= 6) -> Lách rộng 75px
                # Khối 5-7cm (nhỏ hơn, pt_count 3-5) -> Lách vừa 55px
                offset_val = self.DODGE_OFFSET_PX if pt_count >= 6 else self.DODGE_OFFSET_SMALL_PX

                if closest_angle is not None and closest_angle >= 0.0:
                    self.dodge_dir = -1.0
                    self.target_offset_px = -offset_val
                else:
                    self.dodge_dir = 1.0
                    self.target_offset_px = offset_val

                if HAS_ROS:
                    rospy.loginfo(f"[V3 FSM] >>> PHÁT HIỆN KHỐI! Lách {'TRÁI' if self.dodge_dir==-1.0 else 'PHẢI'} {offset_val}px (dist={min_dist:.2f}m)")

        # 2. DODGING -> REENTERING
        elif self.obs_state == FSMObstacleState.DODGING:
            time_in_dodge = now - self.dodge_start_time
            MIN_DODGE_DURATION = 1.2  # Khóa giữ né ít nhất 1.2s để chiều dài 30cm của xe vượt qua khối

            if self.check_side_clearance():
                self.clear_side_count += 1
            else:
                self.clear_side_count = 0

            # Bắt buộc: Phải giữ né đủ 1.2s VÀ hông xe thông thoáng liên tục 8 frame
            is_clear = (time_in_dodge >= MIN_DODGE_DURATION) and (self.clear_side_count >= 8)
            is_timeout = (time_in_dodge > 4.0)

            if is_clear or is_timeout:
                self.obs_state = FSMObstacleState.REENTERING
                self.reenter_start_time = now
                self.target_offset_px = 0.0
                if HAS_ROS:
                    rospy.loginfo(f"[V3 FSM] >>> THÂN XE ĐÃ THOÁT KHỐI TOÀN BỘ ({time_in_dodge:.2f}s) -> BẮT ĐẦU TRẢ LÀN MƯỢT")

        # 3. REENTERING -> NORMAL
        elif self.obs_state == FSMObstacleState.REENTERING:
            if abs(self.current_offset_px) < 1.0 and (now - self.reenter_start_time >= 1.2):
                self.obs_state = FSMObstacleState.NORMAL
                self.dodge_dir = 0.0
                if HAS_ROS:
                    rospy.loginfo("[V3 FSM] HOÀN THÀNH TRẢ LÀN -> VỀ NORMAL")

        # Ramping Offset mượt mà
        step = self.RAMP_STEP_DODGE if self.obs_state == FSMObstacleState.DODGING else self.RAMP_STEP_RETURN
        if self.current_offset_px < self.target_offset_px:
            self.current_offset_px = min(self.target_offset_px, self.current_offset_px + step)
        elif self.current_offset_px > self.target_offset_px:
            self.current_offset_px = max(self.target_offset_px, self.current_offset_px - step)

        return min_dist

    # ==================================================================
    # CORE PIPELINE — Process one frame
    # ==================================================================
    def process_frame(self, frame):
        cfg = self.cfg

        if frame.shape[:2] != (cfg.image_height, cfg.image_width):
            frame = cv2.resize(frame, (cfg.image_width, cfg.image_height))

        # 1. Cập nhật FSM Né Khối 5cm - 10cm
        min_dist = self.update_obstacle_fsm()

        # 2. Undistortion
        frame_undist = self.undistorter.process(frame)

        # 3. ROI crop & Color segmentation
        roi_y_start = int(cfg.roi_y_start * cfg.image_height)
        roi_frame = frame_undist.copy()
        roi_frame[:roi_y_start, :] = 0
        mask = self.segmenter.process(roi_frame)

        # 4. BEV transform
        bev_mask = self.bev.warp_to_bev(mask)

        # 5. Lane detection
        detection = self.lane_detector.detect(bev_mask)

        # 6. Geometry validation
        prev_state = self.state_estimator.state
        observation = self.geometry.process(detection, prev_state)

        # 7. Temporal state estimation
        lane_state = self.state_estimator.update(observation)

        # 8. Trajectory generation
        traj = self.trajectory_gen.generate(lane_state, self.current_speed)

        # 9. Lateral controller (Stanley / Pure Pursuit)
        if cfg.stanley_enabled and lane_state.tracking_state == TrackingState.TRACKING:
            steer_raw = self.stanley_controller.compute(
                lane_state.heading_error,
                lane_state.lateral_error_m,
                lane_state.curvature,
                self.current_speed
            )
        else:
            if traj.target is not None:
                steer_raw = self.pp_controller.compute(traj.target, traj.lookahead_m)
            else:
                steer_raw = 0.0

        # TÍCH HỢP DODGE OFFSET TỪ LIDAR VÀO BẺ LÁI V3 (Đã sửa chiều vô lăng thực tế)
        real_dodge_dir = -self.dodge_dir if (self.cfg and getattr(self.cfg, 'steer_invert', False)) else self.dodge_dir
        if abs(self.current_offset_px) > 0.1:
            steer_offset = (abs(self.current_offset_px) / 120.0) * real_dodge_dir
            steer_raw += steer_offset

            # Safety Steering Override cho khối 10cm (Tối thiểu >= 0.35)
            if self.obs_state == FSMObstacleState.DODGING and abs(steer_raw) < 0.35:
                steer_raw = 0.35 * real_dodge_dir

        # 10. Steering filter
        steer_filtered = self.steering_filter.filter(steer_raw)

        # 10.5. Area Heuristic (CHỈ KÍCH HOẠT KHI Ở TRẠNG THÁI NORMAL ĐỂ KHÔNG CHỐNG LẠI NÉ KHỐI)
        if self.obs_state == FSMObstacleState.NORMAL and lane_state.centerline_poly is not None:
            bottom_y = self.cfg.image_height
            center_x_bottom = int(np.polyval(lane_state.centerline_poly, bottom_y))
            center_x_bottom = max(0, min(self.cfg.image_width, center_x_bottom))
            
            area_left = np.count_nonzero(bev_mask[:, :center_x_bottom])
            area_right = np.count_nonzero(bev_mask[:, center_x_bottom:])
            
            if area_right > area_left * 1.75:
                steer_filtered -= 0.3
            elif area_left > area_right * 1.75:
                steer_filtered += 0.3
                
            steer_filtered = max(-1.0, min(1.0, steer_filtered))

        # 11. Speed controller (Rà phanh khi rẽ / lách khối)
        if self.obs_state == FSMObstacleState.DODGING:
            throttle = 0.15
        else:
            throttle = self.speed_controller.compute(
                traj.curvature,
                lane_state.confidence,
                lane_state.tracking_state,
                actual_speed=None
            )

        # 12. Debug visualization
        dashboard = self.visualizer.render(
            frame, bev_mask, detection, lane_state, traj,
            steer_raw, steer_filtered, throttle,
            self.current_fps
        )

        # 13. HUD Telemetry Overlay & Visual Progress Bar
        now = time.time()
        time_in_dodge = (now - self.dodge_start_time) if self.obs_state == FSMObstacleState.DODGING else 0.0
        
        if dashboard is not None:
            # HUD Overlay
            cv2.putText(dashboard, f"STATE: {self.obs_state} | TIME: {time_in_dodge:.2f}s / {self.MIN_DODGE_DURATION:.1f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(dashboard, f"FRONT: {min_dist:.2f}m | OFFSET: {self.current_offset_px:.0f}px | STEER: {steer_filtered:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Thanh Progress Bar trực quan hiển thị tiến trình né
            if self.obs_state == FSMObstacleState.DODGING:
                progress = min(1.0, time_in_dodge / self.MIN_DODGE_DURATION)
                bar_w = int(200 * progress)
                cv2.rectangle(dashboard, (20, 75), (220, 90), (50, 50, 50), -1)
                cv2.rectangle(dashboard, (20, 75), (20 + bar_w, 90), (0, 255, 0), -1)
                cv2.rectangle(dashboard, (20, 75), (220, 90), (255, 255, 255), 1)

            vis_frame = cv2.resize(dashboard, (640, 480))
            set_web_frame(vis_frame)

        # 14. Ghi Nhật ký CSV Telemetry Từng Frame (30 Hz)
        if self._telemetry_csv:
            self._telemetry_csv.writerow([
                f"{now:.3f}", self.obs_state, f"{time_in_dodge:.2f}", f"{min_dist:.2f}", "0.0",
                f"{self.dodge_dir:.1f}", f"{self.current_offset_px:.1f}", f"{steer_raw:.3f}", f"{steer_filtered:.3f}", f"{throttle:.2f}"
            ])

        # 15. Logging
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

    def run_ros(self):
        print("=== SPEED RACING V3 MASTER (ROS Mode + 5-10cm Cube Avoidance) ===")
        time.sleep(2)
        print("Starting...")

        rate = rospy.Rate(self.cfg.loop_rate)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            frame = self.latest_image.copy()
            steer, throttle, dashboard, lane_state = self.process_frame(frame)

            steer_out = -steer if self.cfg.steer_invert else steer

            if lane_state.tracking_state == TrackingState.E_STOP:
                continue
            else:
                if hasattr(self.racer, 'steer'):
                    self.racer.steer(steer_out, throttle)
                elif hasattr(self.racer, 'set_steering'):
                    self.racer.set_steering(steer_out)
                    if hasattr(self.racer, 'set_throttle'):
                        self.racer.set_throttle(throttle)

            if self.video_writer and dashboard is not None:
                self.video_writer.write(dashboard)

            rate.sleep()

        self._cleanup()

    def _cleanup(self):
        self.racer.stop()
        if self.logger:
            self.logger.close()
        if self.video_writer:
            self.video_writer.release()

def main():
    parser = argparse.ArgumentParser(description='Speed Racing V3 Master')
    parser.add_argument('--video', type=str, default=None, help='Path to video file for offline replay')
    args = parser.parse_args()

    config = V3Config() if V3Config else None
    app = SpeedRacingV3(config=config, video_path=args.video)

    if HAS_ROS:
        try:
            app.run_ros()
        except rospy.ROSInterruptException:
            pass
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            app.racer.stop()

if __name__ == '__main__':
    main()
