#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed Track Simple - JetRacer (Ackermann Steering)
Chỉ bám line, KHÔNG tránh vật cản. Tối ưu tốc độ cho đường hình tròn.
Sa bàn: Đường màu xanh đậm, line màu đỏ cam.
"""
import sys
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

import os, time, math, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import rospy, cv2, numpy as np
from enum import Enum
from sensor_msgs.msg import Image
from src.core.control.racer_controller import RacerController

class TrackState(Enum):
    WAITING = 0
    RACING = 1
    RECOVERING = 2
    CHECKPOINT_CD = 3
    FINISHED = 4

class SimpleSpeedTrack:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO SPEED TRACK SIMPLE (Chỉ bám line) ===")

        # --- Kích thước ảnh ---
        self.W, self.H = 300, 300

        # --- Tốc độ (ưu tiên nhanh) ---
        self.BASE_SPEED = 0.28          # Tốc độ thẳng
        self.CURVE_SPEED = 0.20         # Tốc độ khi cua
        self.RECOVER_SPEED = 0.15       # Tốc độ khi tìm lại line
        self.RECOVER_TIMEOUT = 2.5      # Timeout tìm line (giây)

        # --- PID (lái) ---
        self.Kp = 0.018
        self.Ki = 0.0
        self.Kd = 0.004
        self._pid_integral = 0.0
        self._pid_prev_err = 0.0
        self._pid_last_t = None

        # --- Ngưỡng cua: nếu error > ngưỡng này → giảm tốc ---
        self.CURVE_ERROR_THRESH = 40  # pixel

        # --- ROI (Region of Interest) ---
        self.ROI_Y = int(self.H * 0.75)
        self.ROI_H = int(self.H * 0.20)

        self.LOOK_Y = int(self.H * 0.50)
        self.LOOK_H = int(self.H * 0.15)

        self.FOCUS_WIDTH_PERCENT = 1.0

        # --- Phát hiện line ĐỎ CAM trên nền XANH ĐẬM (HSV) ---
        self.RED_LOWER_1 = np.array([0, 80, 80])
        self.RED_UPPER_1 = np.array([18, 255, 255])
        self.RED_LOWER_2 = np.array([155, 80, 80])
        self.RED_UPPER_2 = np.array([180, 255, 255])

        self.MIN_CONTOUR_AREA = 80

        # --- Checkpoint ---
        self.CP_WHITE_RATIO = 0.45
        self.CP_ROI_Y = int(self.H * 0.88)
        self.CP_ROI_H = int(self.H * 0.10)
        self.cp_count = 0
        self.cp_last_time = 0.0
        self.CP_COOLDOWN_SEC = 3.0

        # --- Hardware ---
        self.racer = RacerController()
        self.racer.stop()

        # --- ROS ---
        self.latest_image = None
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._cam_cb)

        # --- State ---
        self.state = TrackState.WAITING
        self.state_time = rospy.get_time()
        self.WAIT_TIMEOUT = 30.0
        self.LOOP_RATE = 20

        # --- Logger ---
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        self.log_path = os.path.join(log_dir, f'{script_name}_{ts}.csv')
        self._log_file = open(self.log_path, 'w', newline='')
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(['timestamp', 'state', 'steer', 'speed', 'err_px', 'event'])

        self.video_path = os.path.join(log_dir, f'{script_name}_{ts}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 20.0, (self.W, self.H))

        rospy.loginfo("=== SAN SANG ===")

    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                self.latest_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
        except Exception as e:
            rospy.logerr_throttle(5, f"Cam err: {e}")

    def _record_frame(self, frame):
        if self.video_writer is not None and frame is not None:
            if frame.shape[0] != self.H or frame.shape[1] != self.W:
                frame = cv2.resize(frame, (self.W, self.H))
            self.video_writer.write(frame)

    def detect_line(self, frame):
        resized = cv2.resize(frame, (self.W, self.H))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        roi_near = thresh[self.ROI_Y:self.ROI_Y + self.ROI_H, :]
        roi_far = thresh[self.LOOK_Y:self.LOOK_Y + self.LOOK_H, :]

        cnts_near = cv2.findContours(roi_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_near = cnts_near[0] if len(cnts_near) == 2 else cnts_near[1]

        target_x = None
        has_line = False

        if cnts_near:
            largest = max(cnts_near, key=cv2.contourArea)
            if cv2.contourArea(largest) >= self.MIN_CONTOUR_AREA:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx_near = int(M["m10"] / M["m00"])
                    has_line = True
                    target_x = cx_near

                    cnts_far = cv2.findContours(roi_far, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_far = cnts_far[0] if len(cnts_far) == 2 else cnts_far[1]
                    if cnts_far:
                        largest_far = max(cnts_far, key=cv2.contourArea)
                        if cv2.contourArea(largest_far) >= self.MIN_CONTOUR_AREA:
                            M_far = cv2.moments(largest_far)
                            if M_far["m00"] > 0:
                                cx_far = int(M_far["m10"] / M_far["m00"])
                                target_x = int(cx_near * 0.6 + cx_far * 0.4)

        dbg = resized.copy()
        cv2.rectangle(dbg, (0, self.ROI_Y), (self.W, self.ROI_Y + self.ROI_H), (0, 255, 0), 1)
        cv2.rectangle(dbg, (0, self.LOOK_Y), (self.W, self.LOOK_Y + self.LOOK_H), (255, 255, 0), 1)

        if has_line and target_x is not None:
            y_vis = self.ROI_Y + self.ROI_H // 2
            cv2.circle(dbg, (target_x, y_vis), 6, (0, 0, 255), -1)
            cv2.line(dbg, (self.W // 2, self.H), (target_x, y_vis), (0, 255, 0), 2)

        return target_x, has_line, dbg

    def detect_checkpoint(self, image):
        if image is None: return False
        roi = image[self.CP_ROI_Y:self.CP_ROI_Y + self.CP_ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return (np.sum(b > 0) / b.size) >= self.CP_WHITE_RATIO

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

    def set_state(self, s):
        if self.state != s:
            rospy.loginfo(f"STATE: {self.state.name} -> {s.name}")
            self.state = s; self.state_time = rospy.get_time(); self.pid_reset()

    def time_in_state(self):
        return rospy.get_time() - self.state_time

    def log_row(self, steer=0, speed=0, err=0, event=''):
        self._csv.writerow([f'{time.time():.3f}', self.state.name, f'{steer:.3f}', f'{speed:.2f}', f'{err}', event])

    def run(self):
        rospy.loginfo("Đợi 3s..."); time.sleep(3)
        self.log_row(event='RUN_START')
        rate = rospy.Rate(self.LOOP_RATE)

        while not rospy.is_shutdown():
            debug_frame = self.latest_image

            if self.state == TrackState.WAITING:
                self.racer.stop()
                if self.latest_image is not None:
                    _, has_line, _ = self.detect_line(self.latest_image)
                    if has_line:
                        self.set_state(TrackState.RACING)
                elif self.time_in_state() > self.WAIT_TIMEOUT:
                    self.set_state(TrackState.FINISHED)

            elif self.state == TrackState.RACING:
                if self.latest_image is None:
                    self.racer.stop(); rate.sleep(); continue

                target_x, has_line, debug_frame = self.detect_line(self.latest_image)

                if not has_line:
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep(); continue

                err_px = target_x - (self.W / 2.0)
                steer = self.pid_compute(err_px)

                speed = self.CURVE_SPEED if abs(err_px) > self.CURVE_ERROR_THRESH else self.BASE_SPEED
                self.racer.steer(steer, speed)

                if self.detect_checkpoint(self.latest_image):
                    now = time.time()
                    if now - self.cp_last_time >= self.CP_COOLDOWN_SEC:
                        self.cp_count += 1; self.cp_last_time = now
                        rospy.loginfo(f"*** CHECKPOINT {self.cp_count} ***")
                        self.set_state(TrackState.CHECKPOINT_CD)

                self.log_row(steer=steer, speed=speed, err=int(err_px))

            elif self.state == TrackState.RECOVERING:
                if self.latest_image is not None:
                    _, has_line, debug_frame = self.detect_line(self.latest_image)
                    if has_line:
                        self.set_state(TrackState.RACING)
                        rate.sleep(); continue
                self.racer.steer(0.0, self.RECOVER_SPEED)
                if self.time_in_state() > self.RECOVER_TIMEOUT:
                    self.set_state(TrackState.FINISHED)

            elif self.state == TrackState.CHECKPOINT_CD:
                if self.latest_image is None:
                    self.racer.stop(); rate.sleep(); continue

                target_x, has_line, debug_frame = self.detect_line(self.latest_image)
                if not has_line:
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep(); continue

                err_px = target_x - (self.W / 2.0)
                steer = self.pid_compute(err_px)
                speed = self.CURVE_SPEED if abs(err_px) > self.CURVE_ERROR_THRESH else self.BASE_SPEED
                self.racer.steer(steer, speed)

                if self.time_in_state() > 2.0:
                    self.set_state(TrackState.RACING)

            elif self.state == TrackState.FINISHED:
                self.racer.stop(); self.log_row(event='FINISHED'); break

            if debug_frame is not None:
                self._record_frame(debug_frame)

            rate.sleep()

        self.racer.stop()
        self._log_file.close()
        if self.video_writer is not None: self.video_writer.release()
        rospy.loginfo("Kết thúc Simple Speed Track.")

def main():
    rospy.init_node('speed_track_simple', anonymous=True)
    try: SimpleSpeedTrack().run()
    except rospy.ROSInterruptException: pass
    except Exception as e: rospy.logerr(f"Lỗi: {e}"); RacerController().stop()

if __name__ == '__main__': main()
