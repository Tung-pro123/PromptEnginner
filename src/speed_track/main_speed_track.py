#!/usr/bin/env python3
"""
Speed Track Controller - JetRacer ROS AI Kit
Cuộc thi Jetson AI Racer Challenge 2026

Kiến trúc HFSM 3 tầng (Hierarchical Finite State Machine):
  Tầng 1 (Perception): LaneDetector + ObstacleDetector + CheckpointTracker
  Tầng 2 (Decision):   HighLevelState FSM chọn hành vi tối ưu
  Tầng 3 (Execution):  PID Controller + Motor output + CSV Logger

Tham chiếu:
  - Bài báo: s43684-021-00015-x.pdf (HFSM + Energy Efficiency Function)
  - Đề bài: docs/Đề bài chi tiết.docx.pdf (Speed Track rules)
  - Phần cứng: JetRacer ROS AI Kit (Waveshare) - JetBot compatible I2C control
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import rospy
import cv2
import numpy as np
import time
from enum import Enum

from sensor_msgs.msg import LaserScan, Image

# === CÁC MODULE ĐÃ TÁCH ===
from src.core.control.pid_controller import PIDController
from src.core.perception.lane_detector import LaneDetector
from src.core.perception.obstacle_detector import ObstacleDetector
from src.core.perception.checkpoint_tracker import CheckpointTracker
from src.core.utils.csv_logger import CSVLogger


# ============================================================
# TẦNG 2: HFSM - Các trạng thái hành vi (Behavior States)
# Áp dụng từ bài báo Section 2.1 & 2.5 (Upper + Lower Layer)
# ============================================================
class SpeedTrackState(Enum):
    """Máy trạng thái cho bài Speed Track.
    
    Đơn giản hóa từ HFSM 3 tầng trong bài báo:
    - Upper layer scenario → WAITING / RACING / FINISHED
    - Middle layer behavior → KEEP_LANE / AVOID_OBSTACLE
    - Lower layer action → handled by PID controller
    """
    WAITING_FOR_START = 0   # Chờ hiệu lệnh / tìm line
    KEEP_LANE = 1           # Bám lane bình thường (Free driving state)
    AVOID_OBSTACLE = 2      # Né vật cản (Lane change behavior)
    RECOVERING_LANE = 3     # Tìm lại lane sau khi né
    CHECKPOINT_COOLDOWN = 4 # Vừa qua checkpoint, tiếp tục chạy
    EMERGENCY_STOP = 5      # Lỗi nghiêm trọng, dừng xe
    FINISHED = 6            # Hoàn thành lượt chạy


class SpeedTrackController:
    """Controller chính cho bài Speed Track.
    
    Phần cứng: JetRacer ROS AI Kit (Waveshare)
    - Tương thích JetBot I2C (address 0x60)
    - DC encoded motors with PID speed control (RP2040 sub-controller)
    - Điều khiển: robot.set_motors(left_speed, right_speed)
    """

    def __init__(self):
        rospy.loginfo("=== KHỞI TẠO SPEED TRACK CONTROLLER (HFSM) ===")

        self._setup_parameters()
        self._init_hardware()
        self._init_perception()
        self._init_control()
        self._init_logging()

        # === ROS Subscribers ===
        self.latest_image = None
        self.latest_scan = None
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._camera_cb)
        rospy.Subscriber('/scan', LaserScan, self._lidar_cb)

        # === HFSM State ===
        self.current_state = None
        self.state_change_time = rospy.get_time()
        self._set_state(SpeedTrackState.WAITING_FOR_START, initial=True)

        rospy.loginfo("=== KHỞI TẠO HOÀN TẤT. SẴN SÀNG. ===")

    # ----------------------------------------------------------
    # CẤU HÌNH
    # ----------------------------------------------------------
    def _setup_parameters(self):
        """Tham số điều chỉnh được (calibrate trên xe thật)."""
        # Kích thước ảnh xử lý
        self.WIDTH, self.HEIGHT = 300, 300

        # Tốc độ motor
        self.BASE_SPEED = 0.28        # Tốc độ bám lane
        self.AVOID_SPEED = 0.22       # Tốc độ khi né vật cản
        self.RECOVER_SPEED = 0.20     # Tốc độ khi tìm lại lane

        # Thời gian (giây)
        self.AVOID_DURATION = 1.2       # Thời gian tối đa né vật cản
        self.RECOVER_TIMEOUT = 3.0      # Timeout tìm lại lane
        self.CHECKPOINT_COOLDOWN = 2.0  # Cooldown sau checkpoint
        self.WAIT_TIMEOUT = 30.0        # Timeout chờ line ban đầu

        # PID parameters (tune trên xe thật)
        self.PID_KP = 0.45
        self.PID_KI = 0.0
        self.PID_KD = 0.08
        self.PID_OUTPUT_LIMIT = 0.15   # Max motor speed adjustment

        # Avoid offset: khi né vật cản, dịch target sang trái/phải bao nhiêu pixel
        self.AVOID_OFFSET_PIXELS = 80

        # Checkpoint detection: ngưỡng phát hiện vạch checkpoint
        # (vạch ngang sáng chiếm phần lớn chiều rộng ROI)
        self.CHECKPOINT_WHITE_RATIO = 0.45  # 45% ROI là trắng = checkpoint
        self.CHECKPOINT_ROI_Y = None  # Sẽ tính trong _init_perception

        # Vòng lặp
        self.LOOP_RATE = 25  # Hz (target > 20 FPS theo đề bài)

    def _init_hardware(self):
        """Khởi tạo phần cứng JetRacer (JetBot compatible)."""
        try:
            from jetbot import Robot
            self.robot = Robot()
            rospy.loginfo("JetBot Robot hardware initialized.")
        except Exception as e:
            rospy.logwarn(f"JetBot hardware not found, using Mock. Error: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def _init_perception(self):
        """Khởi tạo các module nhận thức (Tầng 1 HFSM)."""
        self.lane_detector = LaneDetector(self.WIDTH, self.HEIGHT)
        self.obstacle_detector = ObstacleDetector(
            safety_distance=0.35,
            warning_distance=0.55,
            min_obstacle_points=5
        )
        self.checkpoint_tracker = CheckpointTracker(cooldown_seconds=3.0)

        # ROI cho checkpoint detection (vùng dưới cùng ảnh)
        self.CHECKPOINT_ROI_Y = int(self.HEIGHT * 0.88)
        self.CHECKPOINT_ROI_H = int(self.HEIGHT * 0.10)

    def _init_control(self):
        """Khởi tạo bộ điều khiển (Tầng 3 HFSM)."""
        self.pid = PIDController(
            kp=self.PID_KP, ki=self.PID_KI, kd=self.PID_KD,
            output_min=-self.PID_OUTPUT_LIMIT,
            output_max=self.PID_OUTPUT_LIMIT
        )
        # Biến theo dõi hướng né
        self.avoid_direction = 'none'

    def _init_logging(self):
        """Khởi tạo hệ thống log CSV (theo đề bài Section 7)."""
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        self.logger = CSVLogger(log_dir=log_dir, prefix='speed_track')
        rospy.loginfo(f"CSV Logger: {self.logger.log_path}")

    # ----------------------------------------------------------
    # ROS CALLBACKS
    # ----------------------------------------------------------
    def _camera_cb(self, msg):
        """Callback xử lý ảnh từ camera CSI."""
        try:
            if 'compressed' in msg.encoding:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1)
            if 'rgb' in msg.encoding:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            rospy.logerr_throttle(5, f"Camera error: {e}")

    def _lidar_cb(self, msg):
        """Callback lưu dữ liệu LiDAR."""
        self.latest_scan = msg

    # ----------------------------------------------------------
    # HFSM STATE MANAGEMENT
    # ----------------------------------------------------------
    def _set_state(self, new_state, initial=False):
        """Chuyển trạng thái FSM (có log).
        
        Áp dụng từ bài báo Section 2.5: State Transition Matrix.
        """
        if self.current_state != new_state:
            if not initial:
                rospy.loginfo(f"STATE: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

            # Reset PID khi chuyển trạng thái để tránh tích lũy sai số cũ
            self.pid.reset()

    def _time_in_state(self):
        """Thời gian (giây) đã ở trong trạng thái hiện tại."""
        return rospy.get_time() - self.state_change_time

    # ----------------------------------------------------------
    # TẦNG 1: PERCEPTION HELPERS
    # ----------------------------------------------------------
    def _detect_checkpoint(self, image):
        """Phát hiện vạch checkpoint bằng camera.
        
        Checkpoint trên sa bàn: vạch ngang trắng/sáng chiếm phần lớn chiều rộng lane.
        """
        if image is None:
            return False

        roi = image[self.CHECKPOINT_ROI_Y:self.CHECKPOINT_ROI_Y + self.CHECKPOINT_ROI_H, :]
        # Chuyển sang grayscale và threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        white_ratio = np.sum(binary > 0) / binary.size
        return white_ratio >= self.CHECKPOINT_WHITE_RATIO

    # ----------------------------------------------------------
    # TẦNG 3: MOTOR CONTROL
    # ----------------------------------------------------------
    def _drive_with_pid(self, line_center_x, base_speed=None):
        """Điều khiển bám lane bằng PID.
        
        Args:
            line_center_x: Tọa độ X trọng tâm line (pixels)
            base_speed: Tốc độ nền (None = dùng BASE_SPEED)
        """
        if base_speed is None:
            base_speed = self.BASE_SPEED

        # Error: dương = line lệch phải, âm = line lệch trái
        error = (line_center_x - self.WIDTH / 2) / (self.WIDTH / 2)  # Normalize to [-1, 1]
        adjustment = self.pid.compute(error)

        left_speed = base_speed + adjustment
        right_speed = base_speed - adjustment

        self.robot.set_motors(left_speed, right_speed)
        return left_speed, right_speed

    def _drive_avoid(self, line_center_x, direction):
        """Điều khiển né vật cản bằng offset tạm thời.
        
        Áp dụng từ bài báo Section 2.3: Lane change behavior.
        Thay vì chuyển làn thực sự, ta dịch target point sang bên.
        """
        offset = -self.AVOID_OFFSET_PIXELS if direction == 'left' else self.AVOID_OFFSET_PIXELS
        target_x = (line_center_x + offset) if line_center_x is not None else (self.WIDTH / 2 + offset)

        error = (target_x - self.WIDTH / 2) / (self.WIDTH / 2)
        adjustment = self.pid.compute(error)

        left_speed = self.AVOID_SPEED + adjustment
        right_speed = self.AVOID_SPEED - adjustment

        self.robot.set_motors(left_speed, right_speed)
        return left_speed, right_speed

    def _stop(self):
        """Dừng motor."""
        self.robot.stop()

    # ----------------------------------------------------------
    # VÒNG LẶP CHÍNH (Main Loop)
    # ----------------------------------------------------------
    def run(self):
        """Vòng lặp chính của Speed Track Controller."""
        rospy.loginfo("Đợi 3 giây trước khi bắt đầu...")
        time.sleep(3)
        rospy.loginfo("=== BẮT ĐẦU LƯỢT CHẠY SPEED TRACK ===")
        self.logger.log_event('RUN_START')

        rate = rospy.Rate(self.LOOP_RATE)

        while not rospy.is_shutdown():
            frame_start = time.time()

            # ============================================
            # STATE 0: CHỜ TÌM LINE ĐỂ BẮT ĐẦU
            # ============================================
            if self.current_state == SpeedTrackState.WAITING_FOR_START:
                self._stop()
                if self.latest_image is not None and self.lane_detector.is_line_visible(self.latest_image):
                    rospy.loginfo("Đã tìm thấy line! Bắt đầu chạy.")
                    self.logger.log_event('LINE_FOUND')
                    self._set_state(SpeedTrackState.KEEP_LANE)
                elif self._time_in_state() > self.WAIT_TIMEOUT:
                    rospy.logerr("Timeout: không tìm thấy line.")
                    self._set_state(SpeedTrackState.EMERGENCY_STOP)

            # ============================================
            # STATE 1: BÁM LANE (Free Driving - bài báo)
            # ============================================
            elif self.current_state == SpeedTrackState.KEEP_LANE:
                if self.latest_image is None:
                    self._stop()
                    rate.sleep()
                    continue

                # --- Tầng 1: Perception ---
                line_cx = self.lane_detector.get_execution_center(self.latest_image)
                obs_result = self.obstacle_detector.analyze(self.latest_scan)
                is_checkpoint = self._detect_checkpoint(self.latest_image)

                # --- Tầng 2: Decision (Energy Efficiency Evaluation) ---
                # Ưu tiên 1: Checkpoint
                if is_checkpoint:
                    cp = self.checkpoint_tracker.try_register_checkpoint()
                    if cp['registered']:
                        rospy.loginfo(f"*** CHECKPOINT {cp['checkpoint_number']} PASSED! ***")
                        self.logger.log_event('CHECKPOINT_PASSED', f"CP{cp['checkpoint_number']}")
                        if cp['all_passed']:
                            rospy.loginfo("=== TẤT CẢ CHECKPOINT ĐÃ VƯỢT ===")
                            self.logger.log_event('ALL_CHECKPOINTS_PASSED')
                        self._set_state(SpeedTrackState.CHECKPOINT_COOLDOWN)
                        rate.sleep()
                        continue

                # Ưu tiên 2: Vật cản (Safety U2 evaluation)
                if obs_result['obstacle_detected'] and obs_result['danger_level'] in ('warning', 'danger'):
                    self.avoid_direction = obs_result['avoid_direction']
                    rospy.loginfo(f"VẬT CẢN! D={obs_result['front_distance']:.2f}m, né {self.avoid_direction}")
                    self.logger.log_event('OBSTACLE_DETECTED',
                                         f"dist={obs_result['front_distance']:.2f},dir={self.avoid_direction}")
                    self._set_state(SpeedTrackState.AVOID_OBSTACLE)
                    rate.sleep()
                    continue

                # Ưu tiên 3: Bám lane bình thường
                if line_cx is not None:
                    ls, rs = self._drive_with_pid(line_cx)
                    latency = (time.time() - frame_start) * 1000
                    self.logger.log(detected_object='lane', decision='keep_lane',
                                   latency_ms=latency, control_output=f'L={ls:.3f},R={rs:.3f}')
                else:
                    # Mất line: kiểm tra ROI xa
                    look_cx = self.lane_detector.get_lookahead_center(self.latest_image)
                    if look_cx is not None:
                        self._drive_with_pid(look_cx, base_speed=self.RECOVER_SPEED)
                    else:
                        rospy.logwarn("Mất line cả 2 ROI! Chuyển sang RECOVERING.")
                        self.logger.log_event('LANE_LOST')
                        self._set_state(SpeedTrackState.RECOVERING_LANE)

            # ============================================
            # STATE 2: NÉ VẬT CẢN (Lane Change - bài báo)
            # ============================================
            elif self.current_state == SpeedTrackState.AVOID_OBSTACLE:
                line_cx = self.lane_detector.get_execution_center(self.latest_image) if self.latest_image is not None else None
                ls, rs = self._drive_avoid(line_cx, self.avoid_direction)

                latency = (time.time() - frame_start) * 1000
                self.logger.log(detected_object='obstacle', decision=f'avoid_{self.avoid_direction}',
                               latency_ms=latency, control_output=f'L={ls:.3f},R={rs:.3f}')

                # Kiểm tra vật cản đã qua chưa
                obs_result = self.obstacle_detector.analyze(self.latest_scan)
                if not obs_result['obstacle_detected'] or obs_result['danger_level'] == 'safe':
                    rospy.loginfo("Đã né qua vật cản. Tìm lại lane.")
                    self.logger.log_event('OBSTACLE_CLEARED')
                    self._set_state(SpeedTrackState.RECOVERING_LANE)
                elif self._time_in_state() > self.AVOID_DURATION:
                    rospy.logwarn("Timeout né vật cản. Thử tìm lại lane.")
                    self._set_state(SpeedTrackState.RECOVERING_LANE)

            # ============================================
            # STATE 3: TÌM LẠI LANE
            # ============================================
            elif self.current_state == SpeedTrackState.RECOVERING_LANE:
                if self.latest_image is not None:
                    line_cx = self.lane_detector.get_execution_center(self.latest_image)
                    if line_cx is not None:
                        rospy.loginfo("Đã tìm lại lane!")
                        self.logger.log_event('LANE_REACQUIRED')
                        self._set_state(SpeedTrackState.KEEP_LANE)
                        rate.sleep()
                        continue

                # Đi thẳng chậm trong khi tìm line
                self.robot.set_motors(self.RECOVER_SPEED, self.RECOVER_SPEED)

                if self._time_in_state() > self.RECOVER_TIMEOUT:
                    rospy.logerr("Timeout tìm lane! Dừng khẩn cấp.")
                    self._set_state(SpeedTrackState.EMERGENCY_STOP)

            # ============================================
            # STATE 4: COOLDOWN SAU CHECKPOINT
            # ============================================
            elif self.current_state == SpeedTrackState.CHECKPOINT_COOLDOWN:
                # Tiếp tục bám lane bình thường trong thời gian cooldown
                if self.latest_image is not None:
                    line_cx = self.lane_detector.get_execution_center(self.latest_image)
                    if line_cx is not None:
                        self._drive_with_pid(line_cx)

                if self._time_in_state() > self.CHECKPOINT_COOLDOWN:
                    self._set_state(SpeedTrackState.KEEP_LANE)

            # ============================================
            # STATE 5 & 6: KẾT THÚC
            # ============================================
            elif self.current_state == SpeedTrackState.EMERGENCY_STOP:
                rospy.logerr("EMERGENCY STOP!")
                self._stop()
                self.logger.log_event('EMERGENCY_STOP')
                break

            elif self.current_state == SpeedTrackState.FINISHED:
                rospy.loginfo("=== HOÀN THÀNH LƯỢT CHẠY ===")
                self._stop()
                cp_status = self.checkpoint_tracker.get_status()
                avg_fps = self.logger.get_average_fps()
                rospy.loginfo(f"Checkpoint: {cp_status['passed']}/{cp_status['total']} ({cp_status['score']} điểm)")
                rospy.loginfo(f"FPS trung bình: {avg_fps:.1f}")
                self.logger.log_event('FINISHED', f"CP={cp_status['score']},FPS={avg_fps:.1f}")
                break

            rate.sleep()

        self._cleanup()

    def _cleanup(self):
        """Giải phóng tài nguyên."""
        rospy.loginfo("Dừng robot và giải phóng tài nguyên...")
        self._stop()

        if hasattr(self, 'logger'):
            avg_fps = self.logger.get_average_fps()
            rospy.loginfo(f"FPS trung bình pipeline: {avg_fps:.1f}")
            fps_bonus = "ĐẠT (10 điểm)" if avg_fps >= 20 else "KHÔNG ĐẠT (0 điểm)"
            rospy.loginfo(f"Tiêu chí FPS >= 20: {fps_bonus}")
            self.logger.close()
            rospy.loginfo(f"Log đã lưu: {self.logger.log_path}")

        rospy.loginfo("Chương trình kết thúc.")


def main():
    rospy.init_node('speed_track_controller', anonymous=True)
    try:
        controller = SpeedTrackController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node bị ngắt.")
    except Exception as e:
        rospy.logerr(f"Lỗi không xác định: {e}", exc_info=True)


if __name__ == '__main__':
    main()