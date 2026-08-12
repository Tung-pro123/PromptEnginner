#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed Track - COMPETITION MODE (Tối đa tốc độ)
- Khử APF (tránh vật cản) để tập trung 100% CPU cho xử lý ảnh.
- Điều tốc động tuyến tính (Max ga ở đường thẳng, tự động hãm mượt mà vào cua).
- Lookahead động và Racing Line (Cắt cua Apex).
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
    WAITING = 0; RACING = 1
    RECOVERING = 3; CHECKPOINT_CD = 4; FINISHED = 6

class CompetitionSpeedController:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO SPEED TRACK COMPETITION (RACING MODE) ===")
        self.W, self.H = 300, 300
        
        # --- TỐC ĐỘ CỰC ĐẠI (RACING MODE) ---
        self.MAX_SPEED = 0.50           # Tốc độ thẳng (Vít kịch kim)
        self.MIN_CORNER_SPEED = 0.22    # Tốc độ tối thiểu để qua cua gắt an toàn
        self.SPEED_DROP_RATE = 0.45     # Hệ số hãm phanh khi vô lăng bẻ
        self.RECOVER_SPEED = 0.18
        
        # --- PID RACING (Chống lắc lư ở tốc độ cao) ---
        self.Kp = 0.022
        self.Ki = 0.000
        self.Kd = 0.012  # D cực lớn để "giảm xóc", dập tắt dao động ngay lập tức
        self._pid_integral = 0.0; self._pid_prev_err = 0.0; self._pid_last_t = None
        self.last_steer = 0.0

        # --- LỌC MÀU HSV (Đỏ Cam) ---
        self.RED_LOWER_1 = np.array([0, 80, 80])
        self.RED_UPPER_1 = np.array([18, 255, 255])
        self.RED_LOWER_2 = np.array([155, 80, 80])
        self.RED_UPPER_2 = np.array([180, 255, 255])
        
        # Checkpoint
        self.CP_WHITE_RATIO = 0.45
        self.CP_ROI_Y = int(self.H * 0.88)
        self.CP_ROI_H = int(self.H * 0.10)
        self.cp_count = 0; self.cp_last_time = 0.0
        self.CP_COOLDOWN_SEC = 3.0
        
        self.WAIT_TIMEOUT = 30.0
        self.RECOVER_TIMEOUT = 2.0
        self.LOOP_RATE = 30 # Đẩy Loop Rate lên 30 để phản ứng cực nhanh

        # --- Hardware & ROS ---
        self.racer = RacerController(); self.racer.stop()
        self.latest_image = None
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._cam_cb, queue_size=1)
        
        self.state = TrackState.WAITING
        self.state_time = rospy.get_time()
        
        # --- Logger ---
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        self.log_path = os.path.join(log_dir, f'{script_name}_{ts}.csv')
        self._log_file = open(self.log_path, 'w', newline='')
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(['timestamp','state','steer','speed','event'])
        self._frame_count = 0; self._fps_start = time.time()
        
        self.video_path = os.path.join(log_dir, f'{script_name}_{ts}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 20.0, (self.W, self.H))
        
        rospy.loginfo("=== SAN SANG XUAT PHAT ===")

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

    def detect_lane(self, frame):
        resized = cv2.resize(frame, (self.W, self.H))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        y_n = int(self.H * 0.70)
        y_m = int(self.H * 0.55)
        y_f = int(self.H * 0.40)

        def find_outer_borders(y):
            pts = np.where(thresh[y, :] == 255)[0]
            if len(pts) == 0:
                return 0, self.W - 1, False
            return pts[0], pts[-1], True

        def get_robust_mid(L, R):
            if R - L < 30:
                if L > self.W // 2: return max(self.W // 4, L - 120)
                else: return min(self.W * 3 // 4, R + 120)
            return (L + R) // 2

        L_n, R_n, has_line = find_outer_borders(y_n)
        mid_n = get_robust_mid(L_n, R_n)
        L_m, R_m, _ = find_outer_borders(y_m)
        mid_m = get_robust_mid(L_m, R_m)
        L_f, R_f, _ = find_outer_borders(y_f)
        mid_f = get_robust_mid(L_f, R_f)

        # --- DYNAMIC LOOKAHEAD & RACING LINE ---
        w_f, w_m, w_n = 0.60, 0.30, 0.10
        if abs(self.last_steer) > 0.5:
            w_f, w_m, w_n = 0.40, 0.40, 0.20
            
        target_x = mid_n * w_n + mid_m * w_m + mid_f * w_f
        
        curve_diff = mid_f - mid_n
        if abs(curve_diff) > 15:
            target_x += curve_diff * 0.35
            
        dbg = resized.copy()
        cv2.line(dbg, (0, y_n), (self.W, y_n), (100, 100, 100), 1)
        cv2.line(dbg, (0, y_f), (self.W, y_f), (100, 100, 100), 1)
        cv2.circle(dbg, (int(mid_n), y_n), 4, (0, 255, 0), -1)
        cv2.circle(dbg, (int(mid_f), y_f), 4, (0, 255, 0), -1)
        
        cv2.circle(dbg, (int(target_x), y_m), 8, (0, 255, 255), 2)
        cv2.line(dbg, (self.W//2, self.H), (int(target_x), y_m), (0, 255, 255), 2)
        
        return target_x, has_line, dbg

    def detect_checkpoint(self, image):
        if image is None: return False
        roi = image[self.CP_ROI_Y:self.CP_ROI_Y+self.CP_ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return (np.sum(b > 0) / b.size) >= self.CP_WHITE_RATIO

    def pid_reset(self):
        self._pid_integral = 0.0; self._pid_prev_err = 0.0; self._pid_last_t = None

    def pid_compute(self, error_px):
        now = time.time()
        dt = 0.033 if self._pid_last_t is None else max(now - self._pid_last_t, 0.01)
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

    def log_row(self, steer=0, speed=0, event=''):
        self._csv.writerow([f'{time.time():.3f}', self.state.name, f'{steer:.3f}', f'{speed:.2f}', event])
        self._frame_count += 1
        if self._frame_count % 30 == 0: self._log_file.flush()

    def run(self):
        rospy.loginfo("Đợi 3s..."); time.sleep(3)
        self.log_row(event='RUN_START')
        rate = rospy.Rate(self.LOOP_RATE)

        while not rospy.is_shutdown():
            debug_frame = self.latest_image

            if self.state == TrackState.WAITING:
                self.racer.stop()
                if self.latest_image is not None:
                    _, has_line, _ = self.detect_lane(self.latest_image)
                    if has_line:
                        self.set_state(TrackState.RACING)
                elif self.time_in_state() > self.WAIT_TIMEOUT:
                    self.set_state(TrackState.FINISHED)

            elif self.state == TrackState.RACING:
                if self.latest_image is None:
                    self.racer.stop(); rate.sleep(); continue

                target_x, has_line, debug_frame = self.detect_lane(self.latest_image)
                
                if not has_line:
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep(); continue

                cam_steer = self.pid_compute(target_x - self.W / 2.0)
                final_steer = max(-1.0, min(1.0, cam_steer))
                self.last_steer = final_steer

                target_speed = self.MAX_SPEED - self.SPEED_DROP_RATE * abs(final_steer)
                target_speed = max(self.MIN_CORNER_SPEED, target_speed)

                self.racer.steer(final_steer, target_speed)

                if self.detect_checkpoint(self.latest_image):
                    now = time.time()
                    if now - self.cp_last_time >= self.CP_COOLDOWN_SEC:
                        self.cp_count += 1; self.cp_last_time = now
                        rospy.loginfo(f"*** CHECKPOINT {self.cp_count} ***")
                        self.set_state(TrackState.CHECKPOINT_CD)
                        
                cv2.putText(debug_frame, f"V:{target_speed:.2f} S:{final_steer:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                self.log_row(steer=final_steer, speed=target_speed)

            elif self.state == TrackState.RECOVERING:
                if self.latest_image is not None:
                    _, has_line, debug_frame = self.detect_lane(self.latest_image)
                    if has_line:
                        self.set_state(TrackState.RACING)
                        rate.sleep(); continue
                self.racer.steer(0.0, self.RECOVER_SPEED)
                if self.time_in_state() > self.RECOVER_TIMEOUT:
                    self.set_state(TrackState.FINISHED)

            elif self.state == TrackState.CHECKPOINT_CD:
                if self.latest_image is None:
                    self.racer.stop(); rate.sleep(); continue

                target_x, has_line, debug_frame = self.detect_lane(self.latest_image)
                if not has_line:
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep(); continue

                cam_steer = self.pid_compute(target_x - self.W / 2.0)
                final_steer = max(-1.0, min(1.0, cam_steer))
                self.last_steer = final_steer
                target_speed = self.MAX_SPEED - self.SPEED_DROP_RATE * abs(final_steer)
                target_speed = max(self.MIN_CORNER_SPEED, target_speed)
                self.racer.steer(final_steer, target_speed)
                
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
        rospy.loginfo("Kết thúc Racing Mode.")

def main():
    rospy.init_node('speed_track_comp', anonymous=True)
    try: CompetitionSpeedController().run()
    except rospy.ROSInterruptException: pass
    except Exception as e: rospy.logerr(f"Lỗi: {e}"); RacerController().stop()

if __name__ == '__main__': main()
