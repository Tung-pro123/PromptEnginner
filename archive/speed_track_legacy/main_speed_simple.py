#!/usr/bin/env python3
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

# ============================================================
# ENUMS
# ============================================================
class TrackState(Enum):
    WAITING = 0
    RACING = 1
    RECOVERING = 2
    CHECKPOINT_CD = 3
    FINISHED = 4

# ============================================================
# MAIN CONTROLLER
# ============================================================
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
        # ROI chính (gần xe) - bám line chính xác
        self.ROI_Y = int(self.H * 0.75)
        self.ROI_H = int(self.H * 0.20)

        # ROI dự báo (xa hơn) - phát hiện sớm cua
        self.LOOK_Y = int(self.H * 0.50)
        self.LOOK_H = int(self.H * 0.15)

        # Focus mask: chỉ lấy 70% giữa ảnh
        self.FOCUS_WIDTH_PERCENT = 1.0

        # --- Phát hiện line ĐỎ CAM trên nền XANH ĐẬM (HSV) ---
        # Đỏ cam: Hue 0-15 hoặc 160-180, Saturation cao, Value cao
        self.RED_LOWER_1 = np.array([0, 80, 80])
        self.RED_UPPER_1 = np.array([18, 255, 255])
        self.RED_LOWER_2 = np.array([155, 80, 80])
        self.RED_UPPER_2 = np.array([180, 255, 255])

        self.MIN_CONTOUR_AREA = 80  # Diện tích tối thiểu contour

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

        # --- CSV Logger ---
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        self.log_path = os.path.join(log_dir, f'{script_name}_{ts}.csv')
        self._log_file = open(self.log_path, 'w', newline='')
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(['timestamp', 'state', 'steer', 'speed', 'error', 'event'])
        self._frame_count = 0
        self._fps_start = time.time()

        # --- Video Logger ---
        self.video_path = os.path.join(log_dir, f'{script_name}_{ts}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.LOOP_RATE, (self.W, self.H))
        if self.video_writer.isOpened():
            rospy.loginfo(f"Video log: {self.video_path}")
        else:
            rospy.logerr("Không thể mở video writer")
            self.video_writer = None

        rospy.loginfo("=== SAN SANG ===")

    # ============================================================
    # ROS CALLBACK
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

    # ============================================================
    # PHÁT HIỆN LINE ĐỎ CAM
    # ============================================================
    def _detect_red_orange_mask(self, roi):
        """Tạo binary mask cho line đỏ cam trên nền xanh đậm."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Đỏ cam nằm ở 2 đầu của dải Hue (0 và 180 nối nhau)
        mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological: loại noise, nối đứt khúc
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def _apply_focus_mask(self, mask):
        """Chỉ giữ vùng giữa ảnh, loại bỏ nhiễu mép."""
        h, w = mask.shape
        focus = np.zeros_like(mask)
        center_w = int(w * self.FOCUS_WIDTH_PERCENT)
        start_x = (w - center_w) // 2
        focus[:, start_x:start_x + center_w] = 255
        return cv2.bitwise_and(mask, focus)

    def get_line_center(self, image, roi_y, roi_h):
        """Tìm tọa độ X trọng tâm line đỏ cam trong ROI."""
        if image is None:
            return None

        roi = image[roi_y:roi_y + roi_h, :]
        mask = self._detect_red_orange_mask(roi)
        mask = self._apply_focus_mask(mask)

        contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.MIN_CONTOUR_AREA:
            return None

        M = cv2.moments(largest)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"])
        return None

    def is_line_visible(self, image):
        """Kiểm tra line có xuất hiện trong ROI chính không."""
        return self.get_line_center(image, self.ROI_Y, self.ROI_H) is not None

    # ============================================================
    # PID STEERING
    # ============================================================
    def pid_reset(self):
        self._pid_integral = 0.0
        self._pid_prev_err = 0.0
        self._pid_last_t = None

    def pid_compute(self, error_px):
        now = time.time()
        dt = 0.05 if self._pid_last_t is None else max(now - self._pid_last_t, 0.01)

        p = self.Kp * error_px
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral + error_px * dt))
        i = self.Ki * self._pid_integral
        d = self.Kd * (error_px - self._pid_prev_err) / dt

        self._pid_prev_err = error_px
        self._pid_last_t = now

        return max(-1.0, min(1.0, p + i + d))

    # ============================================================
    # CHECKPOINT
    # ============================================================
    def detect_checkpoint(self, image):
        if image is None:
            return False
        roi = image[self.CP_ROI_Y:self.CP_ROI_Y + self.CP_ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return (np.sum(b > 0) / b.size) >= self.CP_WHITE_RATIO

    def try_checkpoint(self):
        now = time.time()
        if now - self.cp_last_time < self.CP_COOLDOWN_SEC:
            return False
        self.cp_count += 1
        self.cp_last_time = now
        rospy.loginfo(f"*** CHECKPOINT {self.cp_count} ***")
        return True

    # ============================================================
    # STATE MANAGEMENT
    # ============================================================
    def set_state(self, s):
        if self.state != s:
            rospy.loginfo(f"STATE: {self.state.name} -> {s.name}")
            self.state = s
            self.state_time = rospy.get_time()
            self.pid_reset()

    def time_in_state(self):
        return rospy.get_time() - self.state_time

    def log_row(self, steer=0, speed=0, error=0, event=''):
        self._csv.writerow([
            f'{time.time():.3f}', self.state.name,
            f'{steer:.3f}', f'{speed:.2f}', f'{error:.1f}', event
        ])
        self._frame_count += 1
        if self._frame_count % 20 == 0:
            self._log_file.flush()

    # ============================================================
    # DEBUG DRAWING
    # ============================================================
    def draw_debug(self, image, line_center=None, look_center=None, steer=0, speed=0):
        """Vẽ debug lên ảnh: ROI, tâm line, hướng target."""
        if image is None:
            return None
        dbg = image.copy()

        # Vẽ ROI chính (xanh lá đậm)
        cv2.rectangle(dbg, (0, self.ROI_Y),
                      (self.W - 1, self.ROI_Y + self.ROI_H),
                      (0, 100, 0), 1)

        # Vẽ ROI dự báo (xanh lá đậm nhạt hơn)
        cv2.rectangle(dbg, (0, self.LOOK_Y),
                      (self.W - 1, self.LOOK_Y + self.LOOK_H),
                      (0, 70, 0), 1)

        # Vẽ đường tâm ảnh (trắng, đứt nét)
        mid_x = self.W // 2
        for y in range(0, self.H, 10):
            cv2.line(dbg, (mid_x, y), (mid_x, y + 5), (255, 255, 255), 1)

        # Vẽ tâm line ROI chính (đỏ cam)
        if line_center is not None:
            cv2.line(dbg, (line_center, self.ROI_Y),
                     (line_center, self.ROI_Y + self.ROI_H),
                     (0, 69, 255), 2)
            cv2.circle(dbg, (line_center, self.ROI_Y + self.ROI_H // 2),
                       5, (0, 69, 255), -1)

        # Vẽ tâm line ROI dự báo (đỏ cam nhạt)
        if look_center is not None:
            cv2.line(dbg, (look_center, self.LOOK_Y),
                     (look_center, self.LOOK_Y + self.LOOK_H),
                     (0, 120, 255), 1)

        # Vẽ target line từ chân ảnh đến tâm line (đỏ cam)
        if line_center is not None:
            target_y = self.ROI_Y + self.ROI_H // 2
            cv2.line(dbg, (mid_x, self.H), (line_center, target_y), (0, 69, 255), 2)
            cv2.circle(dbg, (line_center, target_y), 6, (0, 69, 255), 2)

        # Info overlay
        mode = self.state.name
        cv2.putText(dbg, f"{mode} | S:{steer:.2f} V:{speed:.2f} CP:{self.cp_count}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return dbg

    def _record_frame(self, frame):
        if self.video_writer is not None and frame is not None:
            if frame.shape[0] != self.H or frame.shape[1] != self.W:
                frame = cv2.resize(frame, (self.W, self.H))
            self.video_writer.write(frame)

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def run(self):
        rospy.loginfo("Đợi 3s...")
        time.sleep(3)
        rospy.loginfo("=== BẮT ĐẦU SPEED TRACK SIMPLE ===")
        self.log_row(event='RUN_START')
        rate = rospy.Rate(self.LOOP_RATE)

        while not rospy.is_shutdown():
            frame = self.latest_image
            debug_frame = frame

            # ======= WAITING =======
            if self.state == TrackState.WAITING:
                self.racer.stop()
                if frame is not None and self.is_line_visible(frame):
                    rospy.loginfo("Line tìm thấy! Bắt đầu chạy.")
                    self.log_row(event='LINE_FOUND')
                    self.set_state(TrackState.RACING)
                elif self.time_in_state() > self.WAIT_TIMEOUT:
                    rospy.logerr("Timeout chờ line!")
                    self.set_state(TrackState.FINISHED)

            # ======= RACING (Chỉ bám line, không tránh vật cản) =======
            elif self.state == TrackState.RACING:
                if frame is None:
                    self.racer.stop()
                    rate.sleep()
                    continue

                # Lấy tâm line
                line_center = self.get_line_center(frame, self.ROI_Y, self.ROI_H)
                look_center = self.get_line_center(frame, self.LOOK_Y, self.LOOK_H)

                if line_center is None:
                    # Mất line → chuyển sang recovery
                    self.log_row(event='LINE_LOST')
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep()
                    continue

                # Tính lái PID
                error = line_center - self.W / 2.0
                steer = self.pid_compute(error)

                # Tốc độ: nhanh khi thẳng, chậm khi cua
                if abs(error) > self.CURVE_ERROR_THRESH:
                    speed = self.CURVE_SPEED
                else:
                    speed = self.BASE_SPEED

                # Gửi lệnh lái
                self.racer.steer(steer, speed)

                # Checkpoint
                if self.detect_checkpoint(frame):
                    if self.try_checkpoint():
                        self.log_row(steer=steer, speed=speed, error=error,
                                     event=f'CP{self.cp_count}')
                        self.set_state(TrackState.CHECKPOINT_CD)
                        debug_frame = self.draw_debug(frame, line_center, look_center, steer, speed)
                        self._record_frame(debug_frame)
                        rate.sleep()
                        continue

                self.log_row(steer=steer, speed=speed, error=error)
                debug_frame = self.draw_debug(frame, line_center, look_center, steer, speed)

            # ======= RECOVERING (Tìm lại line) =======
            elif self.state == TrackState.RECOVERING:
                if frame is not None and self.is_line_visible(frame):
                    rospy.loginfo("Tìm lại line!")
                    self.log_row(event='LINE_RECOVERED')
                    self.set_state(TrackState.RACING)
                    rate.sleep()
                    continue

                # Đi thẳng chậm để tìm line
                self.racer.steer(0.0, self.RECOVER_SPEED)

                if self.time_in_state() > self.RECOVER_TIMEOUT:
                    rospy.logerr("Không tìm lại line! Dừng.")
                    self.set_state(TrackState.FINISHED)

            # ======= CHECKPOINT COOLDOWN =======
            elif self.state == TrackState.CHECKPOINT_CD:
                if frame is None:
                    self.racer.stop()
                    rate.sleep()
                    continue

                # Vẫn bám line trong cooldown
                line_center = self.get_line_center(frame, self.ROI_Y, self.ROI_H)
                look_center = self.get_line_center(frame, self.LOOK_Y, self.LOOK_H)

                if line_center is None:
                    self.log_row(event='LINE_LOST')
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep()
                    continue

                error = line_center - self.W / 2.0
                steer = self.pid_compute(error)
                speed = self.CURVE_SPEED if abs(error) > self.CURVE_ERROR_THRESH else self.BASE_SPEED
                self.racer.steer(steer, speed)

                debug_frame = self.draw_debug(frame, line_center, look_center, steer, speed)

                if self.time_in_state() > 2.0:
                    self.set_state(TrackState.RACING)

            # ======= FINISHED =======
            elif self.state == TrackState.FINISHED:
                self.racer.stop()
                self.log_row(event='FINISHED')
                break

            # Ghi video debug
            if debug_frame is not None:
                self._record_frame(debug_frame)

            rate.sleep()

        # --- CLEANUP ---
        self.racer.stop()
        elapsed = time.time() - self._fps_start
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        rospy.loginfo(f"FPS: {fps:.1f}, Checkpoints: {self.cp_count}")
        self._log_file.close()
        if self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo("Đã lưu video.")
        rospy.loginfo("Kết thúc.")


def main():
    rospy.init_node('speed_track_simple', anonymous=True)
    try:
        SimpleSpeedTrack().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Lỗi: {e}", exc_info=True)
        try:
            RacerController().stop()
        except:
            pass


if __name__ == '__main__':
    main()
