#!/usr/bin/env python3
"""
Speed Track AI Controller - Blackboard Architecture
Hoà trộn mạng Neural (Lane Keeping) với LiDAR FSM (Né vật cản)
"""
import sys
# Hỗ trợ ROS Python path
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

import os, time, math
import torch
import torch.nn as nn
import numpy as np
import cv2
import rospy
from enum import Enum
from sensor_msgs.msg import LaserScan, Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
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
# BLACKBOARD (BẢNG ĐEN)
# ============================================================
class Blackboard:
    def __init__(self):
        # 1. Dữ liệu Cảm Biến
        self.image = None
        self.scan = None
        
        # 2. Đầu ra của Mô đun AI (Vision)
        self.ai_steer = 0.0
        self.ai_throttle = 0.0
        self.ai_valid = False
        
        # 3. Đầu ra của Mô đun LiDAR (Obstacle FSM)
        self.front_dist = 999.0
        self.avoid_state = AvoidState.NORMAL
        self.avoid_dir = 'right'
        self.lidar_offset_angle = 0.0
        self.current_lidar_offset = 0.0 # Giá trị đã được vuốt mượt (S-Curve)
        
        # 4. Đầu ra của Mô đun Checkpoint
        self.checkpoint_detected = False
        
        # 5. Dữ liệu Quản lý Trạng Thái (FSM/Arbiter)
        self.state = TrackState.WAITING
        self.state_time = 0.0
        self.cp_count = 0
        self.cp_last_time = 0.0
        
        # 6. Lệnh Điều Khiển Cuối Cùng
        self.cmd_steer = 0.0
        self.cmd_throttle = 0.0

# ============================================================
# KIẾN TRÚC MẠNG NEURAL (VisionInferenceModel)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 16, latent_dim)
        )
    def forward(self, x):
        return self.conv(x)

class ControlPredictor(nn.Module):
    def __init__(self, latent_dim=128):
        super(ControlPredictor, self).__init__()
        self.steer_branch = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.throttle_branch = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        steer = self.steer_branch(x)
        throttle = self.throttle_branch(x)
        return torch.cat((steer, throttle), dim=1)

class VisionInferenceModel(nn.Module):
    def __init__(self, latent_dim=128):
        super(VisionInferenceModel, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.predictor = ControlPredictor(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.predictor(z)

# ============================================================
# MAIN CONTROLLER (BLACKBOARD ARCHITECTURE)
# ============================================================
class SpeedTrackBlackboardController:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO SPEED TRACK AI (Blackboard) ===")
        self.bb = Blackboard()
        self.racer = RacerController()
        self.racer.stop()
        
        # --- Cấu hình AI ---
        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 32
        
        # --- Cấu hình Xe (Params) ---
        self.AVOID_SPEED = 0.18
        self.RECOVER_SPEED = 0.15
        
        # --- Cấu hình LiDAR FSM ---
        self.TRIGGER_DIST = 0.70
        self.SIDE_CLEAR_DIST = 0.45
        self.DODGE_ANGLE = 0.50     # Góc bù khi né vật cản (+0.5 hoặc -0.5)
        self.RAMP_STEP_ANGLE = 0.05 # Tốc độ vuốt mượt khi chuyển góc bù
        self.LIDAR_OFFSET_DEG = 180.0
        
        # --- Khởi tạo Pytorch ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Sử dụng thiết bị tính toán AI: {self.device}")
        
        self.model = VisionInferenceModel(latent_dim=128).to(self.device)
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'weights', 'vision_autoencoder.pth')
        
        try:
            full_state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(full_state_dict, strict=False)
            self.model.eval()
            rospy.loginfo(f"Đã load trọng số AI thành công!")
        except Exception as e:
            rospy.logerr(f"Không thể load trọng số AI: {e}")
            sys.exit(1)
            
        # --- Đăng ký ROS Subscribers ---
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._cam_cb, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self._lidar_cb, queue_size=1)
        
        self.set_state(TrackState.WAITING)
        rospy.loginfo("=== SAN SANG ===")

    # ============================================================
    # SENSOR CALLBACKS (Ghi dữ liệu lên Blackboard)
    # ============================================================
    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
            self.bb.image = img
        except Exception as e:
            rospy.logerr_throttle(5, f"Lỗi đọc camera: {e}")

    def _lidar_cb(self, msg):
        self.bb.scan = msg

    # ============================================================
    # MODULES WORKERS (Tính toán và cập nhật Blackboard)
    # ============================================================
    
    def update_ai_module(self):
        """Mô-đun Mắt: Xử lý ảnh và ném vào Neural Network"""
        if self.bb.image is None:
            self.bb.ai_valid = False
            return
            
        frame = self.bb.image
        h, w = frame.shape[:2]
        roi_y1 = int(144 * h / 480)
        roi_y2 = h
        roi_w = min(w, 640) 
        
        roi = frame[roi_y1:roi_y2, 0:roi_w]
        img = cv2.resize(roi, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            self.bb.ai_steer = outputs[0, 0].item()
            # Giữ an toàn cho chân ga
            self.bb.ai_throttle = max(0.0, min(1.0, outputs[0, 1].item()))
            self.bb.ai_valid = True

    def _norm_angle(self, deg):
        deg = deg + self.LIDAR_OFFSET_DEG
        return (deg + 180) % 360 - 180

    def _scan_sector(self, a_min, a_max):
        if self.bb.scan is None: return []
        dists = []
        msg = self.bb.scan
        for i, d in enumerate(msg.ranges):
            a = self._norm_angle(math.degrees(msg.angle_min + i * msg.angle_increment))
            if a_min <= a <= a_max and msg.range_min < d < msg.range_max:
                dists.append(d)
        return dists

    def update_lidar_module(self):
        """Mô-đun Tai: Cập nhật FSM né vật cản, quy đổi ra Góc lái bù (Steer Offset)"""
        d_front = self._scan_sector(-15, 15)
        self.bb.front_dist = min(d_front) if d_front else float('inf')
        
        front = self.bb.front_dist
        
        # Xử lý Logic Máy trạng thái Né vật cản (Avoid FSM)
        if self.bb.avoid_state == AvoidState.NORMAL:
            self.bb.lidar_offset_angle = 0.0
            if front < self.TRIGGER_DIST:
                # Quét 2 bên hông để quyết định né trái hay phải
                ld = self._scan_sector(30, 70); lc = min(ld) if ld else float('inf')
                rd = self._scan_sector(-70, -30); rc = min(rd) if rd else float('inf')
                self.bb.avoid_dir = 'left' if (rc < 0.30 and lc > rc) else 'right'
                
                # Gán góc lái bù (offset)
                self.bb.lidar_offset_angle = self.DODGE_ANGLE if self.bb.avoid_dir == 'right' else -self.DODGE_ANGLE
                self.bb.avoid_state = AvoidState.DODGING
                rospy.loginfo(f"VẬT CẢN {front:.2f}m! Né {self.bb.avoid_dir}")

        elif self.bb.avoid_state == AvoidState.DODGING:
            self.bb.lidar_offset_angle = self.DODGE_ANGLE if self.bb.avoid_dir == 'right' else -self.DODGE_ANGLE
            
            check_left = True if self.bb.avoid_dir == 'right' else False
            d_side = self._scan_sector(70, 110) if check_left else self._scan_sector(-110, -70)
            side_dist = min(d_side) if d_side else float('inf')
            
            if side_dist > self.SIDE_CLEAR_DIST:
                self.bb.avoid_state = AvoidState.REENTERING
                self.bb.lidar_offset_angle = 0.0
                rospy.loginfo("Đã vượt vật cản, quay lại lane")

        elif self.bb.avoid_state == AvoidState.REENTERING:
            self.bb.lidar_offset_angle = 0.0
            if abs(self.bb.current_lidar_offset) < 0.05:
                self.bb.avoid_state = AvoidState.NORMAL
                rospy.loginfo("Về lane thành công")

        # Thuật toán S-Curve Ramp: Vuốt mượt góc bù để xe không bị giật vô lăng đột ngột
        diff = self.bb.lidar_offset_angle - self.bb.current_lidar_offset
        if abs(diff) > 0.01:
            step = np.sign(diff) * self.RAMP_STEP_ANGLE
            if abs(step) > abs(diff):
                self.bb.current_lidar_offset = self.bb.lidar_offset_angle
            else:
                self.bb.current_lidar_offset += step
        else:
            self.bb.current_lidar_offset = self.bb.lidar_offset_angle

    def update_checkpoint_module(self):
        """Mô-đun Checkpoint: Tìm vạch trắng ngang đường"""
        if self.bb.image is None: 
            self.bb.checkpoint_detected = False
            return
        
        H = self.bb.image.shape[0]
        CP_ROI_Y = int(H * 0.88)
        CP_ROI_H = int(H * 0.10)
        roi = self.bb.image[CP_ROI_Y:CP_ROI_Y+CP_ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        ratio = (np.sum(b > 0) / b.size)
        self.bb.checkpoint_detected = (ratio >= 0.45)

    # ============================================================
    # ARBITER (Não bộ Phán Xử)
    # ============================================================
    def set_state(self, s):
        if self.bb.state != s:
            rospy.loginfo(f"STATE: {self.bb.state.name} -> {s.name}")
            self.bb.state = s
            self.bb.state_time = rospy.get_time()

    def time_in_state(self):
        return rospy.get_time() - self.bb.state_time

    def run_arbiter(self):
        """Đọc toàn bộ dữ liệu trên Blackboard để ra quyết định Bẻ lái / Chân ga cuối cùng"""
        
        # --- WAITING ---
        if self.bb.state == TrackState.WAITING:
            self.bb.cmd_steer = 0.0
            self.bb.cmd_throttle = 0.0
            if self.bb.ai_valid:
                self.set_state(TrackState.KEEP_LANE)
            elif self.time_in_state() > 30.0:
                self.set_state(TrackState.E_STOP)

        # --- KEEP LANE (AI Dẫn Đường) ---
        elif self.bb.state == TrackState.KEEP_LANE:
            if not self.bb.ai_valid:
                self.bb.cmd_throttle = 0.0
                return
                
            # HÒA TRỘN TÍN HIỆU: Góc lái AI (giữ làn) + Góc lái bù LiDAR (có thể bằng 0)
            final_steer = self.bb.ai_steer + self.bb.current_lidar_offset
            self.bb.cmd_steer = max(-1.0, min(1.0, final_steer))
            self.bb.cmd_throttle = self.bb.ai_throttle
            
            # Kiểm tra chuyển trạng thái Checkpoint
            if self.bb.checkpoint_detected:
                now = time.time()
                if now - self.bb.cp_last_time > 3.0:
                    self.bb.cp_count += 1
                    self.bb.cp_last_time = now
                    rospy.loginfo(f"*** CHECKPOINT {self.bb.cp_count} ***")
                    self.set_state(TrackState.CHECKPOINT_CD)
                    
            # Kiểm tra chuyển trạng thái Né Vật Cản
            if self.bb.avoid_state != AvoidState.NORMAL:
                self.set_state(TrackState.AVOID_OBSTACLE)

        # --- AVOID OBSTACLE (LiDAR Override) ---
        elif self.bb.state == TrackState.AVOID_OBSTACLE:
            # Vẫn cho phép AI giữ làn, nhưng bị cộng dồn 1 góc né rất mạnh (Lidar Offset)
            final_steer = self.bb.ai_steer + self.bb.current_lidar_offset
            self.bb.cmd_steer = max(-1.0, min(1.0, final_steer))
            
            # Cố định tốc độ an toàn khi né
            self.bb.cmd_throttle = self.AVOID_SPEED 
            
            if self.bb.avoid_state == AvoidState.NORMAL:
                self.set_state(TrackState.KEEP_LANE)
            elif self.time_in_state() > 2.5: # Quá giờ né -> Về Recovering
                self.bb.avoid_state = AvoidState.NORMAL
                self.bb.current_lidar_offset = 0.0
                self.set_state(TrackState.RECOVERING)

        # --- RECOVERING ---
        elif self.bb.state == TrackState.RECOVERING:
            if self.bb.ai_valid:
                self.set_state(TrackState.KEEP_LANE)
            else:
                self.bb.cmd_steer = 0.0
                self.bb.cmd_throttle = self.RECOVER_SPEED
                if self.time_in_state() > 3.0:
                    self.set_state(TrackState.E_STOP)

        # --- CHECKPOINT COOLDOWN ---
        elif self.bb.state == TrackState.CHECKPOINT_CD:
            self.bb.cmd_steer = self.bb.ai_steer
            self.bb.cmd_throttle = self.bb.ai_throttle
            if self.time_in_state() > 2.0:
                self.set_state(TrackState.KEEP_LANE)

        # --- E_STOP / FINISHED ---
        elif self.bb.state in [TrackState.E_STOP, TrackState.FINISHED]:
            self.bb.cmd_steer = 0.0
            self.bb.cmd_throttle = 0.0

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def run(self):
        rospy.loginfo("Đợi 3s để hệ thống khởi động...")
        time.sleep(3)
        rospy.loginfo("=== BẮT ĐẦU ĐIỀU KHIỂN (Hòa Trộn AI + LiDAR) ===")
        
        rate = rospy.Rate(20) 
        
        while not rospy.is_shutdown():
            # 1. Các Module chạy độc lập tính toán và ghi kết quả lên Blackboard
            self.update_ai_module()
            self.update_lidar_module()
            self.update_checkpoint_module()
            
            # 2. NÃO BỘ đọc Blackboard để Ra Quyết Định Cuối Cùng
            self.run_arbiter()
            
            # 3. Gửi lệnh xuống Actuator
            self.racer.steer(self.bb.cmd_steer, self.bb.cmd_throttle)
            
            # In ra Terminal mỗi 0.5s để theo dõi
            rospy.loginfo_throttle(0.5, f"[{self.bb.state.name}] Steer: {self.bb.cmd_steer:+.2f} | Thr: {self.bb.cmd_throttle:.2f} | LidarOffset: {self.bb.current_lidar_offset:+.2f}")
            
            if self.bb.state in [TrackState.E_STOP, TrackState.FINISHED]:
                break
                
            rate.sleep()
            
        self.racer.stop()
        rospy.loginfo("Kết thúc.")

def main():
    rospy.init_node('speed_track_ai_controller', anonymous=True)
    try:
        controller = SpeedTrackBlackboardController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Lỗi: {e}", exc_info=True)
    finally:
        try: RacerController().stop()
        except: pass

if __name__ == '__main__':
    main()
