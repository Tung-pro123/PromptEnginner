#!/usr/bin/env python3
"""
Speed Track Racing v2.0 - Curvature-Aware Controller + BEV Pipeline
Lấy cảm hứng từ CA-MPCC (Lyons 2023) và kiến trúc phân tầng, kết hợp mô hình động học xe đạp (bicycle model).
"""

import sys
import os
import cv2
import math
import time
import numpy as np
import csv

# Thêm đường dẫn src để import core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import rospy
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("WARNING: ROS not found. Running in offline/test mode.")

# Import controller (yêu cầu mock nếu chạy offline không có hardware)
try:
    from src.core.control.racer_controller import RacerController
except ImportError:
    print("WARNING: RacerController not found. Mocking...")
    class RacerController:
        def steer(self, angle, speed): pass
        def stop(self): pass

# ==============================================================================
# PIPELINE XỬ LÝ ẢNH (BEV + Polynomial)
# ==============================================================================
class BEVPipeline:
    def __init__(self, width=640, height=480):
        self.W = width
        self.H = height
        
        # 1. Thông số Màu HSV (CẦN CALIBRATE bằng calib_hsv.py)
        # Tạm định nghĩa dải màu đỏ/cam
        self.H1_MIN, self.S1_MIN, self.V1_MIN = 0, 80, 80
        self.H1_MAX = 18
        self.H2_MIN, self.H2_MAX = 155, 180

        # 2. Thông số BEV Transform (CẦN CALIBRATE bằng calib_bev.py)
        # Giả lập 4 điểm tạo thành hình thang trên ảnh gốc 640x480
        self.src_pts = np.float32([
            [100, 480],  # Trái dưới
            [540, 480],  # Phải dưới
            [400, 250],  # Phải trên
            [240, 250]   # Trái trên
        ])
        
        # Đích đến: hình chữ nhật ở giữa không gian BEV
        # 1 pixel trong BEV = 1 đơn vị khoảng cách (VD: 1px = 1mm)
        self.dst_pts = np.float32([
            [self.W * 0.3, self.H],
            [self.W * 0.7, self.H],
            [self.W * 0.7, 0],
            [self.W * 0.3, 0]
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        
        # 3. CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def process(self, frame):
        """
        Input: BGR frame
        Output: (e_lat, e_psi, curvature, bev_viz)
        """
        # Resize để đảm bảo đúng ma trận BEV
        if frame.shape[:2] != (self.H, self.W):
            frame = cv2.resize(frame, (self.W, self.H))
            
        # --- TẦNG 1: PREPROCESSING (CLAHE) ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        bgr_clahe = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
        
        # --- TẦNG 2: FEATURE EXTRACTION (HSV Hybrid) ---
        hsv = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, (self.H1_MIN, self.S1_MIN, self.V1_MIN), (self.H1_MAX, 255, 255))
        mask2 = cv2.inRange(hsv, (self.H2_MIN, self.S1_MIN, self.V1_MIN), (self.H2_MAX, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Nối đứt khúc
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # --- TẦNG 3: BEV TRANSFORM & POLYNOMIAL FIT ---
        bev_mask = cv2.warpPerspective(mask, self.M, (self.W, self.H))
        
        # Tìm các điểm ảnh sáng (trắng) trong mask BEV
        y_idx, x_idx = np.nonzero(bev_mask)
        
        e_lat, e_psi, kappa = 0.0, 0.0, 0.0
        poly_coeff = None
        
        if len(y_idx) > 50:
            # Fit đa thức bậc 2: x = a*y^2 + b*y + c
            # LƯU Ý: Phải fit x theo y vì đường chạy dọc từ dưới lên
            poly_coeff = np.polyfit(y_idx, x_idx, 2)
            a_p, b_p, c_p = poly_coeff[0], poly_coeff[1], poly_coeff[2]
            
            # Tính toán tại vị trí xe (y = đáy ảnh)
            y_car = self.H
            x_line = a_p*(y_car**2) + b_p*y_car + c_p
            
            # 1. Cross-track error (pixel)
            x_center = self.W / 2.0
            e_lat = x_line - x_center
            
            # 2. Heading error (góc tiếp tuyến)
            # dx/dy = 2*a*y + b
            slope = 2 * a_p * y_car + b_p
            e_psi = math.atan(slope)
            
            # 3. Curvature (tại điểm lookahead xa hơn một chút để dự báo)
            y_look = self.H * 0.5 # Giữa ảnh
            slope_look = 2 * a_p * y_look + b_p
            # Công thức curvature: |x''| / (1 + (x')^2)^(3/2)
            # Nhưng để có dấu (cua trái/phải), ta giữ nguyên x''
            kappa = (2 * a_p) / ((1 + slope_look**2)**1.5)
            
        # Vẽ BEV Viz
        bev_viz = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        if poly_coeff is not None:
            # Vẽ đường đa thức
            ploty = np.linspace(0, self.H-1, self.H)
            plotx = poly_coeff[0]*ploty**2 + poly_coeff[1]*ploty + poly_coeff[2]
            
            pts = np.array([np.transpose(np.vstack([plotx, ploty]))], np.int32)
            cv2.polylines(bev_viz, pts, isClosed=False, color=(0,255,255), thickness=3)
            
            # Vẽ điểm tính e_lat
            cv2.circle(bev_viz, (int(x_center), self.H), 5, (0,0,255), -1)
            cv2.circle(bev_viz, (int(x_line), self.H), 5, (0,255,0), -1)
            cv2.line(bev_viz, (int(x_center), self.H), (int(x_line), self.H), (255,255,255), 2)
            
            # Hiển thị thông số
            cv2.putText(bev_viz, f"e_lat: {e_lat:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(bev_viz, f"e_psi: {math.degrees(e_psi):.1f}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(bev_viz, f"kappa: {kappa:.5f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return e_lat, e_psi, kappa, bev_viz


# ==============================================================================
# THUẬT TOÁN ĐIỀU KHIỂN (Stanley + Curvature-Aware Speed)
# ==============================================================================
class CurvatureAwareController:
    def __init__(self):
        # Hệ số khuếch đại Stanley (CẦN TUNE)
        self.k_stanley = 0.15 
        
        # Chiều dài cơ sở xe (khoảng cách trục trước-sau, mét)
        self.wheelbase = 0.16 
        
        # Hằng số quy đổi pixel sang mét (CẦN TUNE dựa vào kích thước BEV)
        self.px_to_m = 0.005 
        
        # Thông số bám đường (CẦN TUNE bằng calib_speed.py)
        self.mu_g = 1.2 * 9.81 # Hệ số ma sát * gia tốc trọng trường
        self.max_throttle = 0.25
        self.min_throttle = 0.15
        
        # Mapping từ speed m/s sang throttle (Giả lập tuyến tính)
        self.speed_to_throttle_factor = 0.3
        
        self.last_steering = 0.0

    def compute(self, e_lat_px, e_psi_rad, kappa_px, current_throttle):
        """
        Tính toán góc lái và ga.
        """
        # Chuyển đổi đơn vị
        e_lat_m = e_lat_px * self.px_to_m
        kappa_m = kappa_px / self.px_to_m  # kappa = 1/R, R đổi sang m -> kappa tăng
        
        # 1. LAYER 1: CURVATURE-AWARE SPEED PROFILER (Dựa trên vật lý ly tâm)
        # v_max = sqrt(mu * g * R) = sqrt(mu * g / |kappa|)
        if abs(kappa_m) > 1e-4:
            v_max = math.sqrt(self.mu_g / abs(kappa_m))
        else:
            v_max = 2.0 # Đường thẳng, đi nhanh
            
        target_throttle = v_max * self.speed_to_throttle_factor
        target_throttle = max(self.min_throttle, min(self.max_throttle, target_throttle))
        
        # Ước tính vận tốc hiện tại (Rất thô, tốt nhất là lấy từ encoder/wheel speed)
        current_v = max(0.1, current_throttle / self.speed_to_throttle_factor)
        
        # 2. LAYER 2: STANLEY STEERING + FEEDFORWARD
        # a) Feedforward dựa trên curvature (Bicycle model)
        delta_ff = math.atan(self.wheelbase * kappa_m)
        
        # b) Stanley correction
        # arctan(k * e_lat / v)
        delta_stanley = math.atan(self.k_stanley * e_lat_m / current_v)
        
        # Tổng hợp
        steering_rad = e_psi_rad + delta_stanley + delta_ff
        
        # Normalize về dải [-1.0, 1.0] của servo
        # Max lái vật lý thường khoảng 30 độ (0.52 rad)
        max_steer_rad = math.radians(30)
        steering_norm = steering_rad / max_steer_rad
        
        steering_norm = max(-1.0, min(1.0, steering_norm))
        
        # Lọc nhè nhẹ tránh giật
        steering_final = 0.7 * steering_norm + 0.3 * self.last_steering
        self.last_steering = steering_final
        
        return steering_final, target_throttle


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
class SpeedRacingV2:
    def __init__(self):
        if HAS_ROS:
            rospy.init_node('speed_racing_v2', anonymous=True)
            rospy.Subscriber('/csi_cam_0/image_raw', Image, self.cam_cb)
            
        self.pipeline = BEVPipeline(width=640, height=480)
        self.controller = CurvatureAwareController()
        self.racer = RacerController()
        
        self.latest_image = None
        self.current_throttle = 0.0
        
        # Logging
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        self.log_path = os.path.join(log_dir, f'{script_name}_{ts}.csv')
        self.log_file = open(self.log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['timestamp', 'e_lat_px', 'e_psi_deg', 'kappa', 'steer', 'throttle'])
        
        self.video_path = os.path.join(log_dir, f'{script_name}_{ts}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 20.0, (1280, 480))
        print(f"Video log BEV: {self.video_path}")
        
    def cam_cb(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
        except Exception as e:
            print(f"Cam error: {e}")

    def run(self):
        print("=== BẮT ĐẦU SPEED RACING V2 (Curvature-Aware) ===")
        if not HAS_ROS:
            print("Đang chạy offline mode. Chờ ảnh từ webcam hoặc script test.")
            return
            
        rate = rospy.Rate(20) # 20 FPS
        
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                frame = self.latest_image.copy()
                
                # 1. Pipeline xử lý ảnh
                e_lat, e_psi, kappa, bev_viz = self.pipeline.process(frame)
                
                # 2. Controller tính toán
                steer, throttle = self.controller.compute(e_lat, e_psi, kappa, self.current_throttle)
                self.current_throttle = throttle
                
                # 3. Xuất lệnh (ĐẢO DẤU STEER VÌ BỊ NGƯỢC SERVO)
                self.racer.steer(-steer, throttle)
                
                # 4. Debug & Log
                self.csv_writer.writerow([time.time(), e_lat, math.degrees(e_psi), kappa, steer, throttle])
                
                # Ghi video Dashboard (Original + BEV)
                if self.video_writer is not None and bev_viz is not None:
                    # 1. Vẽ vùng BEV lên ảnh gốc
                    cv2.polylines(frame, [np.int32(self.pipeline.src_pts)], isClosed=True, color=(255, 0, 0), thickness=2)
                    
                    # 2. Hiển thị quyết định của xe lên ảnh gốc
                    color_steer = (0, 0, 255) if steer > 0.1 else ((255, 0, 0) if steer < -0.1 else (0, 255, 0))
                    cv2.putText(frame, f"Steer: {steer:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_steer, 2)
                    cv2.putText(frame, f"Throttle: {throttle:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    # 3. Gộp 2 ảnh cạnh nhau (HStack)
                    if bev_viz.shape[:2] != (480, 640):
                        bev_viz = cv2.resize(bev_viz, (640, 480))
                    if frame.shape[:2] != (480, 640):
                        frame = cv2.resize(frame, (640, 480))
                        
                    dashboard = np.hstack((frame, bev_viz))
                    self.video_writer.write(dashboard)
                    
            rate.sleep()
            
        self.racer.stop()
        self.log_file.close()
        if self.video_writer is not None:
            self.video_writer.release()


def run_offline_test():
    """Hàm test offline với webcam máy tính nếu không có ROS"""
    print("Khởi chạy Offline Test với Webcam...")
    cap = cv2.VideoCapture(0)
    pipeline = BEVPipeline()
    controller = CurvatureAwareController()
    current_throttle = 0.15
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Test pipeline
        e_lat, e_psi, kappa, bev_viz = pipeline.process(frame)
        
        # Test controller
        steer, throttle = controller.compute(e_lat, e_psi, kappa, current_throttle)
        current_throttle = throttle
        
        # Hiển thị
        cv2.putText(bev_viz, f"CMD: Steer={steer:.2f}, Thr={throttle:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        cv2.imshow('Webcam View', frame)
        cv2.imshow('BEV Pipeline (Offline)', bev_viz)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if HAS_ROS:
        app = SpeedRacingV2()
        app.run()
    else:
        run_offline_test()
