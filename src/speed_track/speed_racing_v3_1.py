#!/usr/bin/env python3
"""
Speed Racing V3.1 (No BEV, Centroid Tracking)
Combines the speed architecture of V3 with the lightweight centroid-based tracking
of the original simple script. Optimizes for high FPS and stability on Jetson Nano.
"""

import sys
import os
import time
import math
import csv
from enum import Enum

# Fix paths for Python 2/3 compatibility in ROS
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

from src.core.control.racer_controller import RacerController
from src.speed_track.config import V3Config

class TrackState(Enum):
    SEARCHING = 0
    TRACKING = 1
    FINISHED = 2

class SpeedRacingV3_1:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO SPEED RACING V3.1 (Non-BEV) ===")
        
        self.cfg = V3Config()
        
        # Hardware
        self.racer = RacerController()
        self.racer.stop()
        
        # ROS
        self.latest_image = None
        rospy.Subscriber(self.cfg.camera_topic, Image, self._cam_cb, queue_size=1)
        
        # State
        self.state = TrackState.SEARCHING
        self.state_time = rospy.get_time()
        
        # Image dimensions (will be updated when first frame arrives)
        self.H = self.cfg.image_height
        self.W = self.cfg.image_width
        self.W_mid = self.W // 2
        self._last_center_x = self.W_mid  # Mỏ neo thời gian (Temporal Anchor)
        
        # PID
        self._pid_integral = 0.0
        self._pid_prev_err = 0.0
        self._pid_last_t = None
        
        # Logging
        self._init_logging()

    def _init_logging(self):
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        
        self.log_path = os.path.join(log_dir, f'v3.1_{ts}.csv')
        self._log_file = open(self.log_path, 'w', newline='')
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(['timestamp', 'fps', 'state', 'center_x', 'error_m', 'steer', 'throttle'])
        
        self.video_writer = None
        if self.cfg.record_video:
            self.video_path = os.path.join(log_dir, f'v3.1_{ts}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.cfg.video_fps, 
                (self.W, self.H)
            )

    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
            
            # Resize to expected dimensions for consistent ROI
            if img.shape[0] != self.H or img.shape[1] != self.W:
                img = cv2.resize(img, (self.W, self.H))
                
            self.latest_image = img
        except Exception as e:
            rospy.logerr_throttle(5, f"Cam err: {e}")

    def _get_hsv_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Mask 1
        lower1 = np.array([self.cfg.hsv_h1_min, self.cfg.hsv_s_min, self.cfg.hsv_v_min])
        upper1 = np.array([self.cfg.hsv_h1_max, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Mask 2 (wrap-around)
        lower2 = np.array([self.cfg.hsv_h2_min, self.cfg.hsv_s_min, self.cfg.hsv_v_min])
        upper2 = np.array([self.cfg.hsv_h2_max, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def _find_multilane_peaks(self, mask, y_start, y_end):
        """
        Nhận diện Vạch Giữa Nét Đứt Đỏ + 2 Vạch Biên Nét Liền Đỏ.
        Nếu Vạch Giữa nét đứt bị khuất (vào khoảng trống gap), tự động tái tạo
        tâm từ 2 vạch biên (left + right) / 2 để không bao giờ bị bám nhầm vào biên!
        """
        roi_mask = mask[y_start:y_end, :]
        histogram = np.sum(roi_mask, axis=0)
        
        min_height = (y_end - y_start) * 255 * 0.08
        peaks = self._find_peaks_1d(histogram, min_height)
        
        left_x, center_x, right_x = None, None, None
        expected_half_track = int(self.W * 0.28)  # Ước lượng nửa chiều rộng đường đua (pixels)
        
        if not peaks:
            return left_x, center_x, right_x
            
        if len(peaks) >= 3:
            # Tìm thấy đủ 3 vạch: [Biên trái, Vạch Giữa Nét Đứt, Biên phải]
            c_idx = min(range(len(peaks)), key=lambda i: abs(peaks[i] - self._last_center_x))
            center_x = peaks[c_idx]
            if c_idx > 0:
                left_x = peaks[c_idx - 1]
            if c_idx < len(peaks) - 1:
                right_x = peaks[c_idx + 1]
        elif len(peaks) == 2:
            p0, p1 = peaks[0], peaks[1]
            gap = p1 - p0
            # Nếu 2 vạch cách nhau khoảng 1 đường đua (~0.35 - 0.85 chiều rộng ảnh)
            if int(self.W * 0.35) <= gap <= int(self.W * 0.85):
                # Đây chính là 2 VẠCH BIÊN (Trái & Phải), vạch nét đứt ở giữa đang vào khoảng trống!
                left_x = p0
                right_x = p1
                center_x = (p0 + p1) // 2  # Tái tạo vạch giữa nét đứt bằng trung điểm 2 biên!
            else:
                # 1 vạch biên + 1 vạch giữa
                c_idx = min(range(2), key=lambda i: abs(peaks[i] - self._last_center_x))
                center_x = peaks[c_idx]
                if c_idx == 0:
                    right_x = peaks[1]
                else:
                    left_x = peaks[0]
        elif len(peaks) == 1:
            p0 = peaks[0]
            # Nếu đỉnh duy nhất gần mỏ neo -> chính là Vạch Giữa
            if abs(p0 - self._last_center_x) < expected_half_track * 0.7:
                center_x = p0
            elif p0 < self._last_center_x:
                # Vạch biên trái -> Tính vạch giữa ước lượng
                left_x = p0
                center_x = p0 + expected_half_track
            else:
                # Vạch biên phải -> Tính vạch giữa ước lượng
                right_x = p0
                center_x = p0 - expected_half_track
                
        if center_x is not None:
            # Làm mượt Mỏ neo thời gian (EMA filter)
            self._last_center_x = int(0.7 * self._last_center_x + 0.3 * center_x)
            
        return left_x, center_x, right_x

    def _find_peaks_1d(self, histogram, min_height):
        min_distance = int(self.W * 0.2)
        peaks = []
        for i in range(1, len(histogram) - 1):
            if histogram[i] < min_height:
                continue
            if histogram[i] > histogram[i-1] and histogram[i] >= histogram[i+1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
                else:
                    if histogram[i] > histogram[peaks[-1]]:
                        peaks[-1] = i
        return peaks

    def _pid_compute(self, error):
        now = time.time()
        dt = 0.05 if self._pid_last_t is None else max(now - self._pid_last_t, 0.01)
        
        # Using PID constants from config, fallback to defaults if not set
        kp = getattr(self.cfg, 'steer_pid_kp', 0.8)
        ki = getattr(self.cfg, 'steer_pid_ki', 0.0)
        kd = getattr(self.cfg, 'steer_pid_kd', 0.05)
        
        p = kp * error
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral + error * dt))
        i = ki * self._pid_integral
        d = kd * (error - self._pid_prev_err) / dt
        
        self._pid_prev_err = error
        self._pid_last_t = now
        
        # Multiply by a factor to convert normalized error to steering angle range
        steer = p + i + d
        return max(-1.0, min(1.0, steer))

    def run(self):
        rospy.loginfo("Waiting for camera...")
        while self.latest_image is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        rospy.loginfo("=== V3.1 BAT DAU ===")
        rate = rospy.Rate(self.cfg.loop_rate)
        
        fps_start = time.time()
        frame_count = 0
        current_fps = 0.0
        
        while not rospy.is_shutdown():
            frame = self.latest_image
            if frame is None:
                rate.sleep()
                continue
                
            frame_start = time.time()
            debug_img = frame.copy()
            
            # Create full HSV mask
            mask = self._get_hsv_mask(frame)
            
            # Define ROI for tracking (e.g. bottom 40% of the image)
            roi_y_start = int(self.H * 0.6)
            roi_y_end = self.H
            
            # Define ROI for lookahead (e.g. middle 20% of the image)
            look_y_start = int(self.H * 0.4)
            look_y_end = int(self.H * 0.6)
            
            # Find peaks
            l_x, c_x, r_x = self._find_multilane_peaks(mask, roi_y_start, roi_y_end)
            look_l, look_c, look_r = self._find_multilane_peaks(mask, look_y_start, look_y_end)
            
            steer_out = 0.0
            throttle_out = 0.0
            error_norm = 0.0
            
            # Strict center tracking: only use center_x. Fallback to lookahead center if needed.
            center_x = c_x
            if center_x is None and look_c is not None:
                center_x = look_c 
                
            if center_x is not None:
                self.state = TrackState.TRACKING
                
                # Normalized error: -1.0 (left) to 1.0 (right)
                target_x = self.W / 2.0
                error_px = center_x - target_x
                error_norm = error_px / (self.W / 2.0)
                
                # Steering (Positive error = line is to the right = steer right)
                steer_raw = self._pid_compute(error_norm)
                steer_out = steer_raw
                
                if self.cfg.steer_invert:
                    steer_out = -steer_out
                    
                # Speed control (slow down if error is large)
                speed_factor = max(0.4, 1.0 - abs(error_norm) * 1.5)
                v_target = self.cfg.cruise_speed * speed_factor
                v_target = max(self.cfg.min_speed, min(self.cfg.max_speed, v_target))
                
                throttle_out = v_target * self.cfg.speed_to_throttle_factor
            else:
                self.state = TrackState.SEARCHING
                # Crawl forward to find line
                throttle_out = self.cfg.min_speed * self.cfg.speed_to_throttle_factor
                steer_out = 0.0
                
            # Draw Multi-Lane Debug
            roi_mid_y = roi_y_start + (roi_y_end - roi_y_start)//2
            look_mid_y = look_y_start + (look_y_end - look_y_start)//2
            
            # Left = Blue, Center = Green, Right = Red
            if l_x is not None: cv2.circle(debug_img, (l_x, roi_mid_y), 6, (255, 0, 0), -1)
            if c_x is not None: cv2.circle(debug_img, (c_x, roi_mid_y), 8, (0, 255, 0), -1)
            if r_x is not None: cv2.circle(debug_img, (r_x, roi_mid_y), 6, (0, 0, 255), -1)
            
            if look_l is not None: cv2.circle(debug_img, (look_l, look_mid_y), 4, (255, 0, 0), -1)
            if look_c is not None: cv2.circle(debug_img, (look_c, look_mid_y), 5, (0, 255, 0), -1)
            if look_r is not None: cv2.circle(debug_img, (look_r, look_mid_y), 4, (0, 0, 255), -1)
                
            # Actuate
            self.racer.steer(steer_out, throttle_out)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_start
                current_fps = 10.0 / elapsed
                fps_start = time.time()
                
            # Draw overlays
            cv2.rectangle(debug_img, (0, roi_y_start), (self.W, roi_y_end), (0, 255, 255), 1)
            cv2.putText(debug_img, f"FPS: {current_fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_img, f"State: {self.state.name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.state == TrackState.TRACKING else (0, 165, 255), 2)
            cv2.putText(debug_img, f"Steer: {steer_out:.2f}  Thr: {throttle_out:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add mask as inset for debugging
            mask_bgr = cv2.cvtColor(cv2.resize(mask, (self.W//4, self.H//4)), cv2.COLOR_GRAY2BGR)
            debug_img[0:self.H//4, self.W - self.W//4:self.W] = mask_bgr
            
            # Log
            self._csv.writerow([
                f'{time.time():.3f}', f'{current_fps:.1f}', self.state.name,
                center_x if center_x else -1, f'{error_norm:.3f}', 
                f'{steer_out:.3f}', f'{throttle_out:.3f}'
            ])
            if frame_count % 20 == 0:
                self._log_file.flush()
                
            if self.video_writer is not None:
                self.video_writer.write(debug_img)
                
            rate.sleep()
            
        # Cleanup
        self.racer.stop()
        self._log_file.close()
        if self.video_writer:
            self.video_writer.release()
        rospy.loginfo("=== KET THUC ===")

if __name__ == '__main__':
    rospy.init_node('speed_racing_v3_1', anonymous=True)
    try:
        SpeedRacingV3_1().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        try:
            RacerController().stop()
        except:
            pass
