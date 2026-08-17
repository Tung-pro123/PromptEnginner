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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import rospy, cv2, numpy as np
from enum import Enum
from sensor_msgs.msg import LaserScan, Image
from src.core.control.racer_controller import RacerController

# ============================================================
# ENUMS
# ============================================================
class TrackState(Enum):
    WAITING = 0; KEEP_LANE = 1
    RECOVERING = 3; CHECKPOINT_CD = 4; E_STOP = 5; FINISHED = 6

# ============================================================
# MAIN CONTROLLER
# ============================================================
class SpeedTrackController:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO SPEED TRACK (Hybrid Lane + Ackermann) ===")
        # --- Params ---
        self.W, self.H = 300, 300
        self.BASE_SPEED = 0.22
        self.AVOID_SPEED = 0.13
        self.RECOVER_SPEED = 0.15
        self.AVOID_TIMEOUT = 3.5
        self.RECOVER_TIMEOUT = 3.0
        self.CP_COOLDOWN = 2.0
        self.WAIT_TIMEOUT = 30.0
        self.LOOP_RATE = 20
        # PID (steering output) - Tăng Kp, Kd để phản hồi cua nhanh hơn
        self.Kp = 0.020; self.Ki = 0.000; self.Kd = 0.005
        self._pid_integral = 0.0; self._pid_prev_err = 0.0; self._pid_last_t = None
        # APF (Artificial Potential Field) - thay thế FSM né vật cản
        self.LIDAR_OFFSET_DEG = 180.0
        self.APF_GAIN = 0.15               # Hệ số lực đẩy
        self.APF_INFLUENCE_DIST = 0.80     # Bán kính ảnh hưởng (m)
        self.APF_FRONTAL_BIAS = 0.3        # Lực bias cho vật cản chính diện
        self.E_STOP_DIST = 0.25            # Phanh khẩn cấp
        self.apf_last_steer = 0.0          # Lưu giá trị APF frame trước
        # Checkpoint
        self.CP_WHITE_RATIO = 0.45
        self.CP_ROI_Y = int(self.H * 0.88)
        self.CP_ROI_H = int(self.H * 0.10)
        self.cp_count = 0; self.cp_last_time = 0.0
        self.CP_COOLDOWN_SEC = 3.0
        # Lane detection params (Áp dụng lọc màu HSV từ simple)
        self.RED_LOWER_1 = np.array([0, 80, 80])
        self.RED_UPPER_1 = np.array([18, 255, 255])
        self.RED_LOWER_2 = np.array([155, 80, 80])
        self.RED_UPPER_2 = np.array([180, 255, 255])
        
        self.CURVE_STEER_THRESH = 0.40    # Ngưỡng đánh lái để nhận diện đang cua
        self.CURVE_SPEED_FACTOR = 0.75    # Giảm 25% tốc độ khi cua
        
        self.BORDER_SAFETY_MARGIN = 0.15  # Giữ 15% khoảng cách tới biên
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

        # --- Video Logger ---
        self.video_path = os.path.join(log_dir, f'speed_{ts}.avi')
        self.video_writer = None
        self.initialize_video_writer()

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

    def initialize_video_writer(self):
        """Khởi tạo VideoWriter để ghi log video."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.LOOP_RATE, (self.W * 2, self.H))
            if self.video_writer.isOpened():
                rospy.loginfo(f"Ghi video debug vào file: {self.video_path}")
            else:
                rospy.logerr("Không thể mở file video để ghi.")
                self.video_writer = None
        except Exception as e:
            rospy.logerr(f"Lỗi khởi tạo VideoWriter: {e}")
            self.video_writer = None

    def _record_frame(self, frame):
        """Ghi khung hình camera + BEV Map Lidar vào video log."""
        if self.video_writer is not None:
            try:
                bev = self.draw_bev_map()
                if frame is not None:
                    if frame.shape[0] != self.H or frame.shape[1] != self.W:
                        frame = cv2.resize(frame, (self.W, self.H))
                else:
                    frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
                
                combined = cv2.hconcat([frame, bev])
                self.video_writer.write(combined)
            except Exception as e:
                rospy.logerr_throttle(5, f"Lỗi ghi video: {e}")

    def draw_bev_map(self):
        """Vẽ bản đồ nhìn từ trên xuống (Bird's Eye View) của Lidar."""
        bev = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        # Grid lines
        for i in range(1, 4): cv2.line(bev, (0, i*100), (self.W, i*100), (40,40,40), 1)
        for i in range(1, 3): cv2.line(bev, (i*100, 0), (i*100, self.H), (40,40,40), 1)

        cx, cy = self.W // 2, int(self.H * 0.85)
        scale = 100.0  # 1m = 100px
        
        # Vehicle marker
        cv2.rectangle(bev, (cx - 10, cy - 20), (cx + 10, cy + 20), (255, 255, 255), -1)
        
        if self.latest_scan:
            msg = self.latest_scan
            for i, d in enumerate(msg.ranges):
                if msg.range_min < d < msg.range_max and d < 1.5:
                    deg = math.degrees(msg.angle_min + i * msg.angle_increment) + self.LIDAR_OFFSET_DEG
                    a = math.radians((deg + 180) % 360 - 180)
                    x, y = d * math.cos(a), d * math.sin(a)
                    
                    px = int(cx - y * scale)
                    py = int(cy - x * scale)
                    if 0 <= px < self.W and 0 <= py < self.H:
                        if d < 0.3: col = (0,0,255)
                        elif d < 0.6: col = (0,165,255)
                        elif d < 1.0: col = (0,255,255)
                        else: col = (0,255,0)
                        cv2.circle(bev, (px, py), 2, col, -1)
                        
        # Info overlays
        cv2.putText(bev, f"APF: {self.apf_last_steer:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(bev, f"State: {self.state.name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
        return bev

    # ============================================================
    # HYBRID LANE DETECTION (Cải tiến)
    # Tìm biên trái và phải bằng cách scan toàn bộ dòng, dùng 3 điểm lookahead
    # ============================================================
    def detect_lane(self, frame):
        """Returns (target_x, left_border, right_border, has_line, debug_img)"""
        resized = cv2.resize(frame, (self.W, self.H))
        
        # [HỌC TỪ main_speed_simple] Lọc màu ĐỎ CAM (HSV) chính xác hơn thay vì Grayscale
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # [HỌC TỪ main_speed_simple] Dùng Morphology loại bỏ nhiễu và nối các vạch đứt khúc
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        y_n = int(self.H * 0.70)  # Near
        y_m = int(self.H * 0.55)  # Mid
        y_f = int(self.H * 0.40)  # Far

        def find_outer_borders(y):
            pts = np.where(thresh[y, :] == 255)[0]
            if len(pts) == 0:
                return 0, self.W - 1, False
            return pts[0], pts[-1], True

        def get_robust_mid(L, R):
            if R - L < 30: # Chỉ thấy 1 vạch (có thể do đứt nét hoặc khuất)
                if L > self.W // 2: return max(self.W // 4, L - 120)
                else: return min(self.W * 3 // 4, R + 120)
            return (L + R) // 2

        L_n, R_n, has_line_n = find_outer_borders(y_n)
        mid_n = get_robust_mid(L_n, R_n)
        
        L_m, R_m, _ = find_outer_borders(y_m)
        mid_m = get_robust_mid(L_m, R_m)
        
        L_f, R_f, _ = find_outer_borders(y_f)
        mid_f = get_robust_mid(L_f, R_f)

        # Target: Blend giữa các điểm để tạo đường cong mượt.
        # Ưu tiên far (0.5) để lookahead ôm cua sớm.
        target_x = mid_n * 0.2 + mid_m * 0.3 + mid_f * 0.5

        # Debug image
        dbg = resized.copy()
        
        # Vẽ các vùng quét ROI
        cv2.line(dbg, (0, y_n), (self.W, y_n), (100, 100, 100), 1)
        cv2.line(dbg, (0, y_m), (self.W, y_m), (100, 100, 100), 1)
        cv2.line(dbg, (0, y_f), (self.W, y_f), (100, 100, 100), 1)
        
        # Vẽ các điểm biên và trung điểm
        cv2.circle(dbg, (L_n, y_n), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (R_n, y_n), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (int(mid_n), y_n), 4, (0, 255, 0), -1)
        
        cv2.circle(dbg, (L_m, y_m), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (R_m, y_m), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (int(mid_m), y_m), 4, (0, 255, 0), -1)
        
        cv2.circle(dbg, (L_f, y_f), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (R_f, y_f), 4, (0, 0, 255), -1)
        cv2.circle(dbg, (int(mid_f), y_f), 4, (0, 255, 0), -1)
        
        # Vẽ đường polyline thể hiện dự đoán quỹ đạo cong
        pts_curve = np.array([[mid_n, y_n], [mid_m, y_m], [mid_f, y_f]], np.int32)
        pts_curve = pts_curve.reshape((-1, 1, 2))
        cv2.polylines(dbg, [pts_curve], False, (0, 255, 255), 2)
        
        cv2.putText(dbg, "MODE: Dynamic Lookahead", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return target_x, L_n, R_n, has_line_n, dbg

    def is_line_visible(self, frame):
        """Check nhanh xem có thấy đường không."""
        try:
            _, _, _, has_line, _ = self.detect_lane(frame)
            return has_line
        except Exception as e:
            rospy.logerr_throttle(2, f"is_line_visible error: {e}")
            return False

    # ============================================================
    # OBSTACLE DETECTION (LiDAR)
    # ============================================================
    def _norm_angle(self, deg):
        deg = deg + self.LIDAR_OFFSET_DEG
        return (deg + 180) % 360 - 180

    def _scan_sector_with_offset(self, a_min, a_max, offset):
        if self.latest_scan is None: return []
        dists = []
        msg = self.latest_scan
        for i, d in enumerate(msg.ranges):
            deg = math.degrees(msg.angle_min + i * msg.angle_increment) + offset
            a = (deg + 180) % 360 - 180
            if a_min <= a <= a_max and msg.range_min < d < msg.range_max:
                dists.append(d)
        return dists

    def _scan_sector(self, a_min, a_max):
        return self._scan_sector_with_offset(a_min, a_max, self.LIDAR_OFFSET_DEG)

    def log_lidar_diagnostics(self):
        if self.latest_scan is None:
            rospy.loginfo_throttle(3, "[LIDAR] Chưa nhận được dữ liệu scan từ topic /scan!")
            return
        
        # Thử quét hướng thẳng với các offset góc khác nhau để dò hướng thật
        d_0 = self._scan_sector_with_offset(-15, 15, offset=0.0)
        d_180 = self._scan_sector_with_offset(-15, 15, offset=180.0)
        d_90 = self._scan_sector_with_offset(-15, 15, offset=90.0)
        d_270 = self._scan_sector_with_offset(-15, 15, offset=270.0)
        
        min_0 = min(d_0) if d_0 else float('inf')
        min_180 = min(d_180) if d_180 else float('inf')
        min_90 = min(d_90) if d_90 else float('inf')
        min_270 = min(d_270) if d_270 else float('inf')
        
        rospy.loginfo_throttle(3, 
            f"\n[LIDAR DIAGNOSTICS] Khoảng cách phía trước vật lý ứng với các Offset:\n"
            f"  - Nếu đầu Lidar hướng thẳng (Offset 0.0): {min_0:.2f}m\n"
            f"  - Nếu Lidar quay ngược 180 độ (Offset 180.0): {min_180:.2f}m\n"
            f"  - Nếu Lidar lệch trái 90 độ (Offset 90.0): {min_90:.2f}m\n"
            f"  - Nếu Lidar lệch phải 90 độ (Offset 270.0): {min_270:.2f}m\n"
            f"  => Đang cấu hình LIDAR_OFFSET_DEG = {self.LIDAR_OFFSET_DEG}"
        )

    def get_filtered_min_dist(self, dists):
        """Lọc nhiễu cảm biến bằng phân vị (percentile filter) tránh nhiễu điểm đơn độc."""
        if not dists: return float('inf')
        sorted_d = sorted(dists)
        # Bỏ qua 10% số điểm nhỏ nhất (tối thiểu bỏ 1 điểm) để chống nhiễu hạt bụi/nhiễu xung
        idx = min(len(sorted_d) - 1, max(1, len(sorted_d) // 10))
        return sorted_d[idx]

    def get_front_dist(self):
        self.log_lidar_diagnostics()
        d = self._scan_sector(-15, 15)
        return self.get_filtered_min_dist(d)

    def compute_apf_steering(self):
        """
        Tính lực lái APF từ LiDAR.
        Returns: (apf_steer, min_front_dist, speed_factor)
        """
        if self.latest_scan is None:
            return 0.0, float('inf'), 1.0

        msg = self.latest_scan
        lateral_force = 0.0
        min_front = float('inf')

        # Xác định bên nào trống hơn (dùng cho vật cản chính diện)
        left_dists = self._scan_sector(30, 70)
        right_dists = self._scan_sector(-70, -30)
        left_space = self.get_filtered_min_dist(left_dists)
        right_space = self.get_filtered_min_dist(right_dists)
        bias_dir = 1.0 if left_space >= right_space else -1.0

        for i, d in enumerate(msg.ranges):
            if not (msg.range_min < d < msg.range_max) or d > self.APF_INFLUENCE_DIST:
                continue

            deg = math.degrees(msg.angle_min + i * msg.angle_increment) + self.LIDAR_OFFSET_DEG
            angle = math.radians((deg + 180) % 360 - 180)
            angle_deg = math.degrees(angle)

            if abs(angle_deg) > 90:
                continue

            if abs(angle_deg) < 20:
                min_front = min(min_front, d)

            # Lực đẩy: càng gần càng mạnh
            force = self.APF_GAIN * (1.0 / d - 1.0 / self.APF_INFLUENCE_DIST)

            if abs(angle_deg) < 15:
                # Vật cản chính diện: sin(≈ 0) không cho lực ngang, thêm bias
                lateral_force += force * self.APF_FRONTAL_BIAS * bias_dir
            else:
                # Vật cản bên: sin(angle) tự nhiên đẩy ra xa
                lateral_force += force * math.sin(angle)

        apf_steer = max(-1.0, min(1.0, lateral_force))

        # Giảm tốc khi gần vật cản
        speed_factor = 1.0
        if min_front < self.APF_INFLUENCE_DIST:
            speed_factor = max(0.4, min_front / self.APF_INFLUENCE_DIST)

        self.apf_last_steer = apf_steer
        return apf_steer, min_front, speed_factor

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
            debug_frame = self.latest_image

            # --- PHANH KHẨN CẤP AN TOÀN (LIDAR Failsafe) ---
            if self.state not in [TrackState.WAITING, TrackState.E_STOP, TrackState.FINISHED]:
                front_dist = self.get_front_dist()
                if front_dist < 0.32:
                    rospy.logerr(f"!!! QUÁ NGUY HIỂM: Vật cản trước mặt cách {front_dist:.2f}m !!! PHANH KHẨN CẤP")
                    self.log_row(event='E_STOP_LIDAR')
                    self.set_state(TrackState.E_STOP)

            # --- WAITING ---
            if self.state == TrackState.WAITING:
                self.racer.stop()
                img_ok = self.latest_image is not None
                vis = self.is_line_visible(self.latest_image) if img_ok else False
                rospy.loginfo_throttle(2, f"[WAITING] Image received: {img_ok}, Line visible: {vis}")
                
                if img_ok and vis:
                    try:
                        _, _, _, _, debug_frame = self.detect_lane(self.latest_image)
                    except:
                        pass
                    self.log_row(event='LINE_FOUND')
                    self.set_state(TrackState.KEEP_LANE)
                elif self.time_in_state() > self.WAIT_TIMEOUT:
                    self.set_state(TrackState.E_STOP)

            # --- KEEP LANE (Hybrid APF) ---
            elif self.state == TrackState.KEEP_LANE:
                if self.latest_image is None:
                    self.racer.stop(); rate.sleep(); continue

                if not self.is_line_visible(self.latest_image):
                    self.log_row(event='LANE_LOST')
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep(); continue

                # 1. Trích xuất Đặc trưng Ngữ nghĩa (Camera Semantic Features)
                target_x, L, R, has_center, debug_frame = self.detect_lane(self.latest_image)
                cam_steer = self.pid_compute(target_x - self.W / 2.0)

                # 2. Trích xuất Đặc trưng Hình học (LiDAR Geometric Features - APF)
                apf_steer, min_front, speed_factor = self.compute_apf_steering()

                # 3. Cơ chế Đánh trọng số Chú ý (Attention-Based Feature Fusion)
                # Giải quyết "Semantic Blindness": LiDAR không biết đâu là lề đường.
                # Camera biết lề đường (L, R). Ta dùng không gian BEV để tính khoảng cách an toàn.
                car_center = self.W / 2.0
                margin = self.W * self.BORDER_SAFETY_MARGIN  # Khoảng cách an toàn tối thiểu tới lề
                
                w_apf = 1.0  # Trọng số chú ý cho LiDAR (Attention Weight)
                
                if apf_steer < 0:
                    # LiDAR muốn đẩy xe sang TRÁI. Kiểm tra lề trái L.
                    dist_to_left = car_center - L
                    if dist_to_left < margin:
                        w_apf = 0.0  # Sát lề quá rồi, không tin LiDAR nữa
                    elif dist_to_left < margin * 2:
                        w_apf = (dist_to_left - margin) / margin # Giảm dần trọng số
                elif apf_steer > 0:
                    # LiDAR muốn đẩy xe sang PHẢI. Kiểm tra lề phải R.
                    dist_to_right = R - car_center
                    if dist_to_right < margin:
                        w_apf = 0.0
                    elif dist_to_right < margin * 2:
                        w_apf = (dist_to_right - margin) / margin
                        
                # 4. Kết hợp lực lái (Decision-Level Fusion)
                # cam_steer luôn được giữ để bám lane, apf_steer bị scale bởi Attention Weight
                final_steer = max(-1.0, min(1.0, cam_steer + w_apf * apf_steer))
                
                # [HỌC TỪ main_speed_simple] Giảm tốc độ khi vào cua để không văng
                curve_factor = self.CURVE_SPEED_FACTOR if abs(final_steer) > self.CURVE_STEER_THRESH else 1.0
                target_speed = self.BASE_SPEED * speed_factor * curve_factor

                # 4. An toàn: Phanh khẩn cấp nếu vật cản quá gần (dù APF đang đẩy)
                if min_front < self.E_STOP_DIST:
                    rospy.logwarn(f"Vật cản ở {min_front:.2f}m - Quá sát! Dừng khẩn cấp!")
                    self.racer.stop()
                    self.log_row(event='APF_ESTOP')
                    self.set_state(TrackState.E_STOP)
                    continue

                # 5. Gửi lệnh điều khiển
                self.racer.steer(final_steer, target_speed)

                # Checkpoint
                if self.detect_checkpoint(self.latest_image):
                    if self.try_checkpoint():
                        self.log_row(event=f'CP{self.cp_count}')
                        self.set_state(TrackState.CHECKPOINT_CD)
                        self._record_frame(debug_frame)
                        rate.sleep(); continue

                self.log_row(steer=final_steer, speed=target_speed, front=min_front, offset=apf_steer)
                if debug_frame is not None:
                    self._draw_target(debug_frame, target_x)

            # --- RECOVERING ---
            elif self.state == TrackState.RECOVERING:
                if self.latest_image is not None and self.is_line_visible(self.latest_image):
                    try:
                        _, _, _, _, debug_frame = self.detect_lane(self.latest_image)
                    except:
                        pass
                    self.log_row(event='LANE_FOUND')
                    self.set_state(TrackState.KEEP_LANE)
                    self._record_frame(debug_frame)
                    rate.sleep(); continue
                self.racer.steer(0.0, self.RECOVER_SPEED)
                if self.time_in_state() > self.RECOVER_TIMEOUT:
                    self.set_state(TrackState.E_STOP)

            # --- CHECKPOINT COOLDOWN ---
            elif self.state == TrackState.CHECKPOINT_CD:
                if self.latest_image is None:
                    self.racer.stop(); rate.sleep(); continue

                if not self.is_line_visible(self.latest_image):
                    self.log_row(event='LANE_LOST')
                    self.set_state(TrackState.RECOVERING)
                    rate.sleep(); continue

                if self.latest_image is not None:
                    target_x, L, R, _, debug_frame = self.detect_lane(self.latest_image)
                    self.steer_to(target_x, self.BASE_SPEED)
                    if debug_frame is not None:
                        self._draw_target(debug_frame, target_x)
                if self.time_in_state() > self.CP_COOLDOWN:
                    self.set_state(TrackState.KEEP_LANE)

            # --- E_STOP / FINISHED ---
            elif self.state == TrackState.E_STOP:
                self.racer.stop(); self.log_row(event='E_STOP'); break
            elif self.state == TrackState.FINISHED:
                self.racer.stop(); self.log_row(event='FINISHED'); break

            if debug_frame is not None:
                self._record_frame(debug_frame)

            rate.sleep()

        # --- CLEANUP (Chạy khi thoát vòng lặp while do bấm Ctrl+C) ---
        self.racer.stop()
        elapsed = time.time() - self._fps_start
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        rospy.loginfo(f"FPS: {fps:.1f}, CP: {self.cp_count}/3")
        self._log_file.close()
        if self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo("Đã lưu và đóng file video.")
        rospy.loginfo("Kết thúc.")

    def _draw_target(self, frame, target_x):
        """Vẽ Target thực sự xe đang hướng tới sau khi tính toán offset."""
        y_near = int(self.H * 0.55)
        cv2.circle(frame, (int(target_x), y_near - 20), 8, (255, 0, 0), 2)
        cv2.line(frame, (self.W//2, self.H), (int(target_x), y_near - 20), (255, 0, 0), 2)
        cv2.putText(frame, "Target", (int(target_x) - 20, y_near - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Cleanup code moved from here to the end of run()


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
