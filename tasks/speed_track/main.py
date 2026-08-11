#!/usr/bin/env python3
"""
Speed Track Controller - JetRacer (Ackermann Steering) - Single File
Hybrid lane detection: Tìm 2 biên trắng + vạch giữa đứt khúc
"""
import sys
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

import os, time, math, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import rospy, cv2, numpy as np
from enum import Enum
from sensor_msgs.msg import LaserScan, Image
from src.core.control.racer_controller import RacerController

# ============================================================
# ENUMS
# ============================================================
class TrackState(Enum):
    WAITING = 0; KEEP_LANE = 1; AVOID_OBSTACLE = 2
    RECOVERING = 3; CHECKPOINT_CD = 4; E_STOP = 5; FINISHED = 6

class AvoidState(Enum):
    NORMAL = 0; DODGING = 1; REENTERING = 2

# ============================================================
# MAIN CONTROLLER
# ============================================================
class SpeedTrackController:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO SPEED TRACK (Hybrid Lane + Ackermann) ===")
        # --- Params ---
        self.W, self.H = 300, 300
        self.BASE_SPEED = 0.22
        self.AVOID_SPEED = 0.18
        self.RECOVER_SPEED = 0.15
        self.AVOID_TIMEOUT = 2.5
        self.RECOVER_TIMEOUT = 3.0
        self.CP_COOLDOWN = 2.0
        self.WAIT_TIMEOUT = 30.0
        self.LOOP_RATE = 20
        # PID (steering output)
        self.Kp = 0.007; self.Ki = 0.0; self.Kd = 0.002
        self._pid_integral = 0.0; self._pid_prev_err = 0.0; self._pid_last_t = None
        # Obstacle FSM
        self.TRIGGER_DIST = 0.70
        self.SIDE_CLEAR_DIST = 0.45
        self.DODGE_OFFSET_PX = 55
        self.RAMP_STEP_PX = 4
        self.LIDAR_OFFSET_DEG = 180.0
        self.avoid_state = AvoidState.NORMAL
        self.target_offset = 0.0; self.current_offset = 0.0
        self.avoid_dir = 'right'
        # Checkpoint
        self.CP_WHITE_RATIO = 0.45
        self.CP_ROI_Y = int(self.H * 0.88)
        self.CP_ROI_H = int(self.H * 0.10)
        self.cp_count = 0; self.cp_last_time = 0.0
        self.CP_COOLDOWN_SEC = 3.0
        # Lane detection params
        self.GRAY_THRESH = 180
        self.BORDER_SAFETY_MARGIN = 0.30  # Giữ 30% khoảng cách tới biên
        # --- Hardware ---
        self.racer = RacerController(); self.racer.stop()
        # --- ROS ---
        self.latest_image = None; self.latest_scan = None
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._cam_cb)
        rospy.Subscriber('/scan', LaserScan, self._lidar_cb)
        # --- State ---
        self.state = TrackState.WAITING
        self.state_time = rospy.get_time()
        # --- CSV Logger ---
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'speed_{ts}.csv')
        self._log_file = open(self.log_path, 'w', newline='')
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(['timestamp','state','steer','speed','front_dist','offset','event'])
        self._frame_count = 0; self._fps_start = time.time()
        rospy.loginfo(f"Log: {self.log_path}")
        rospy.loginfo("=== SAN SANG ===")

    # ============================================================
    # ROS CALLBACKS
    # ============================================================
    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                self.latest_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
        except Exception as e:
            rospy.logerr_throttle(5, f"Cam err: {e}")

    def _lidar_cb(self, msg):
        self.latest_scan = msg

    # ============================================================
    # HYBRID LANE DETECTION (Cách B)
    # Tìm 2 biên trắng + vạch trắng đứt khúc giữa
    # ============================================================
    def detect_lane(self, frame):
        """Returns (target_x, left_border, right_border, has_center_line, debug_img)"""
        resized = cv2.resize(frame, (self.W, self.H))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.GRAY_THRESH, 255, cv2.THRESH_BINARY)

        y_near = int(self.H * 0.85)
        y_far = int(self.H * 0.55)

        def find_borders(y):
            mid = self.W // 2
            L, R = 0, self.W - 1
            for x in range(mid, 0, -1):
                if thresh[y, x] == 255: L = x; break
            for x in range(mid, self.W):
                if thresh[y, x] == 255: R = x; break
            return L, R, (L + R) // 2

        L_n, R_n, mid_n = find_borders(y_near)
        L_f, R_f, mid_f = find_borders(y_far)

        # Tìm vạch trắng đứt khúc giữa bằng contour trong vùng giữa 2 biên
        roi_y = int(self.H * 0.70)
        roi_h = int(self.H * 0.25)
        roi = thresh[roi_y:roi_y+roi_h, :]
        # Mask chỉ giữ vùng giữa 2 biên (loại bỏ biên)
        margin = 15  # pixel margin tránh biên
        mask_roi = np.zeros_like(roi)
        left_safe = min(L_n, L_f) + margin
        right_safe = max(R_n, R_f) - margin
        if left_safe < right_safe:
            mask_roi[:, left_safe:right_safe] = roi[:, left_safe:right_safe]
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center_line_x = None
        has_center = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 60:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    center_line_x = int(M["m10"] / M["m00"])
                    has_center = True

        # Target: ưu tiên vạch giữa, fallback = trung điểm 2 biên
        target_x = center_line_x if has_center else mid_n

        # Debug image
        dbg = resized.copy()
        cv2.line(dbg, (0, y_near), (self.W, y_near), (0, 255, 255), 1)
        cv2.circle(dbg, (L_n, y_near), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (R_n, y_near), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (mid_n, y_near), 5, (255, 0, 0), -1)
        if has_center:
            cv2.circle(dbg, (center_line_x, roi_y + roi_h//2), 5, (0, 255, 0), -1)

        return target_x, L_n, R_n, has_center, dbg

    def is_line_visible(self, frame):
        """Check nhanh xem có thấy đường không."""
        try:
            _, L, R, _, _ = self.detect_lane(frame)
            return (R - L) > 30  # Có khoảng cách hợp lý giữa 2 biên
        except:
            return False

    # ============================================================
    # OBSTACLE DETECTION (LiDAR)
    # ============================================================
    def _norm_angle(self, deg):
        deg = deg + self.LIDAR_OFFSET_DEG
        return (deg + 180) % 360 - 180

    def _scan_sector(self, a_min, a_max):
        if self.latest_scan is None: return []
        dists = []
        msg = self.latest_scan
        for i, d in enumerate(msg.ranges):
            a = self._norm_angle(math.degrees(msg.angle_min + i * msg.angle_increment))
            if a_min <= a <= a_max and msg.range_min < d < msg.range_max:
                dists.append(d)
        return dists

    def get_front_dist(self):
        d = self._scan_sector(-15, 15)
        return min(d) if d else float('inf')

    def is_side_clear(self, side='left'):
        if side == 'left':
            d = self._scan_sector(70, 110)
        else:
            d = self._scan_sector(-110, -70)
        return min(d) > self.SIDE_CLEAR_DIST if d else True

    def choose_avoid_dir(self):
        ld = self._scan_sector(30, 70)
        rd = self._scan_sector(-70, -30)
        lc = min(ld) if ld else float('inf')
        rc = min(rd) if rd else float('inf')
        return 'left' if (rc < 0.30 and lc > rc) else 'right'

    def update_obstacle_fsm(self):
        """Cập nhật FSM né + S-Curve ramp. Returns offset_px."""
        front = self.get_front_dist()

        if self.avoid_state == AvoidState.NORMAL:
            self.target_offset = 0.0
            if front < self.TRIGGER_DIST:
                self.avoid_dir = self.choose_avoid_dir()
                self.target_offset = self.DODGE_OFFSET_PX if self.avoid_dir == 'right' else -self.DODGE_OFFSET_PX
                self.avoid_state = AvoidState.DODGING
                rospy.loginfo(f"VẬT CẢN {front:.2f}m! Né {self.avoid_dir}")

        elif self.avoid_state == AvoidState.DODGING:
            self.target_offset = self.DODGE_OFFSET_PX if self.avoid_dir == 'right' else -self.DODGE_OFFSET_PX
            check = 'left' if self.avoid_dir == 'right' else 'right'
            if self.is_side_clear(check):
                self.avoid_state = AvoidState.REENTERING
                self.target_offset = 0.0
                rospy.loginfo("Đã vượt vật cản, quay lại lane")

        elif self.avoid_state == AvoidState.REENTERING:
            self.target_offset = 0.0
            if abs(self.current_offset) < 1.0:
                self.avoid_state = AvoidState.NORMAL
                rospy.loginfo("Về lane thành công")

        # S-Curve ramp
        diff = self.target_offset - self.current_offset
        if abs(diff) > 0.1:
            step = np.sign(diff) * self.RAMP_STEP_PX
            self.current_offset = self.target_offset if abs(step) > abs(diff) else self.current_offset + step
        else:
            self.current_offset = self.target_offset

        return self.current_offset, front

    def clamp_offset_by_borders(self, offset, left_b, right_b):
        """Giới hạn offset để bánh không ra khỏi vùng đen."""
        center = self.W / 2.0
        max_right = (right_b - center) * (1.0 - self.BORDER_SAFETY_MARGIN)
        max_left = (left_b - center) * (1.0 - self.BORDER_SAFETY_MARGIN)
        return max(max_left, min(max_right, offset))

    # ============================================================
    # PID STEERING
    # ============================================================
    def pid_reset(self):
        self._pid_integral = 0.0; self._pid_prev_err = 0.0; self._pid_last_t = None

    def pid_compute(self, error_px):
        now = time.time()
        dt = 0.05 if self._pid_last_t is None else max(now - self._pid_last_t, 0.01)
        p = self.Kp * error_px
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral + error_px * dt))
        i = self.Ki * self._pid_integral
        d = self.Kd * (error_px - self._pid_prev_err) / dt
        self._pid_prev_err = error_px; self._pid_last_t = now
        return max(-1.0, min(1.0, p + i + d))

    def steer_to(self, target_x, speed=None):
        speed = speed or self.BASE_SPEED
        steering = self.pid_compute(target_x - self.W / 2.0)
        self.racer.steer(steering, speed)
        return steering

    # ============================================================
    # CHECKPOINT
    # ============================================================
    def detect_checkpoint(self, image):
        if image is None: return False
        roi = image[self.CP_ROI_Y:self.CP_ROI_Y+self.CP_ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return (np.sum(b > 0) / b.size) >= self.CP_WHITE_RATIO

    def try_checkpoint(self):
        now = time.time()
        if now - self.cp_last_time < self.CP_COOLDOWN_SEC: return False
        self.cp_count += 1; self.cp_last_time = now
        rospy.loginfo(f"*** CHECKPOINT {self.cp_count} ***")
        return True

    # ============================================================
    # STATE MANAGEMENT
    # ============================================================
    def set_state(self, s):
        if self.state != s:
            rospy.loginfo(f"STATE: {self.state.name} -> {s.name}")
            self.state = s; self.state_time = rospy.get_time(); self.pid_reset()

    def time_in_state(self):
        return rospy.get_time() - self.state_time

    def log_row(self, steer=0, speed=0, front=0, offset=0, event=''):
        self._csv.writerow([f'{time.time():.3f}', self.state.name,
            f'{steer:.3f}', f'{speed:.2f}', f'{front:.2f}', f'{offset:.1f}', event])
        self._frame_count += 1
        if self._frame_count % 20 == 0: self._log_file.flush()

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def run(self):
        rospy.loginfo("Đợi 3s..."); time.sleep(3)
        rospy.loginfo("=== BẮT ĐẦU SPEED TRACK ===")
        self.log_row(event='RUN_START')
        rate = rospy.Rate(self.LOOP_RATE)

        while not rospy.is_shutdown():
            # --- WAITING ---
            if self.state == TrackState.WAITING:
                self.racer.stop()
                if self.latest_image is not None and self.is_line_visible(self.latest_image):
                    self.log_row(event='LINE_FOUND')
                    self.set_state(TrackState.KEEP_LANE)
                elif self.time_in_state() > self.WAIT_TIMEOUT:
                    self.set_state(TrackState.E_STOP)

            # --- KEEP LANE ---
            elif self.state == TrackState.KEEP_LANE:
                if self.latest_image is None:
                    self.racer.stop(); rate.sleep(); continue

                target_x, L, R, has_center, _ = self.detect_lane(self.latest_image)
                offset, front = self.update_obstacle_fsm()

                # Checkpoint
                if self.detect_checkpoint(self.latest_image):
                    if self.try_checkpoint():
                        self.log_row(event=f'CP{self.cp_count}')
                        self.set_state(TrackState.CHECKPOINT_CD)
                        rate.sleep(); continue

                # Chuyển sang AVOID nếu obstacle FSM đang né
                if self.avoid_state != AvoidState.NORMAL:
                    self.set_state(TrackState.AVOID_OBSTACLE)

                # Giới hạn offset theo biên
                safe_offset = self.clamp_offset_by_borders(offset, L, R)
                final_target = target_x + safe_offset
                steer = self.steer_to(final_target, self.BASE_SPEED)
                self.log_row(steer=steer, speed=self.BASE_SPEED, front=front, offset=safe_offset)

            # --- AVOID OBSTACLE ---
            elif self.state == TrackState.AVOID_OBSTACLE:
                offset, front = self.update_obstacle_fsm()
                target_x, L, R = self.W // 2, 0, self.W - 1
                has_center = False
                if self.latest_image is not None:
                    target_x, L, R, has_center, _ = self.detect_lane(self.latest_image)

                safe_offset = self.clamp_offset_by_borders(offset, L, R)
                final_target = target_x + safe_offset
                steer = self.steer_to(final_target, self.AVOID_SPEED)
                self.log_row(steer=steer, speed=self.AVOID_SPEED, front=front, offset=safe_offset)

                if self.avoid_state == AvoidState.NORMAL:
                    self.log_row(event='OBSTACLE_CLEARED')
                    self.set_state(TrackState.KEEP_LANE)
                elif self.time_in_state() > self.AVOID_TIMEOUT:
                    self.avoid_state = AvoidState.NORMAL
                    self.current_offset = 0.0; self.target_offset = 0.0
                    self.set_state(TrackState.RECOVERING)

            # --- RECOVERING ---
            elif self.state == TrackState.RECOVERING:
                if self.latest_image is not None and self.is_line_visible(self.latest_image):
                    self.log_row(event='LANE_FOUND')
                    self.set_state(TrackState.KEEP_LANE)
                    rate.sleep(); continue
                self.racer.steer(0.0, self.RECOVER_SPEED)
                if self.time_in_state() > self.RECOVER_TIMEOUT:
                    self.set_state(TrackState.E_STOP)

            # --- CHECKPOINT COOLDOWN ---
            elif self.state == TrackState.CHECKPOINT_CD:
                if self.latest_image is not None:
                    target_x, L, R, _, _ = self.detect_lane(self.latest_image)
                    self.steer_to(target_x, self.BASE_SPEED)
                if self.time_in_state() > self.CP_COOLDOWN:
                    self.set_state(TrackState.KEEP_LANE)

            # --- E_STOP / FINISHED ---
            elif self.state == TrackState.E_STOP:
                self.racer.stop(); self.log_row(event='E_STOP'); break
            elif self.state == TrackState.FINISHED:
                self.racer.stop(); self.log_row(event='FINISHED'); break

            rate.sleep()

        self.racer.stop()
        elapsed = time.time() - self._fps_start
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        rospy.loginfo(f"FPS: {fps:.1f}, CP: {self.cp_count}/3")
        self._log_file.close()
        rospy.loginfo("Kết thúc.")

def main():
    rospy.init_node('speed_track_controller', anonymous=True)
    try:
        SpeedTrackController().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Lỗi: {e}", exc_info=True)
        try: RacerController().stop()
        except: pass

if __name__ == '__main__':
    main()