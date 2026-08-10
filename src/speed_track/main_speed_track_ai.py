#!/usr/bin/env python3
"""
Speed Track AI Controller - JetRacer (End-to-End Deep Learning)
Trực tiếp nhận ảnh từ camera, dùng Pytorch đưa ra góc lái và ga (chạy trên GPU nếu có)
"""
import sys
# Hỗ trợ ROS Python path
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.control.racer_controller import RacerController

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
        # Nhánh chuyên biệt học góc lái (Steer)
        self.steer_branch = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        # Nhánh chuyên biệt học vận tốc (Throttle)
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
# MAIN AI CONTROLLER
# ============================================================
class SpeedTrackAIController:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO SPEED TRACK AI (End-to-End) ===")
        
        # --- Cấu hình Ảnh ---
        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 32
        
        # --- Khởi tạo xe ---
        self.racer = RacerController()
        self.racer.stop()
        
        # --- Khởi tạo Pytorch Model (chạy trên GPU nếu có) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Sử dụng thiết bị tính toán: {self.device}")
        
        self.model = VisionInferenceModel(latent_dim=128).to(self.device)
        weights_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'experiments', 'weights', 'vision_autoencoder.pth')
        
        try:
            # strict=False cho phép load thành công dù cho file .pth có chứa cả trọng số của lớp Decoder dư thừa
            full_state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(full_state_dict, strict=False)
            self.model.eval()
            rospy.loginfo(f"Đã load trọng số AI thành công!")
        except Exception as e:
            rospy.logerr(f"Không thể load trọng số AI tại {weights_path}: {e}")
            sys.exit(1)
            
        # --- Cấu hình ROS ---
        self.latest_image = None
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._cam_cb, queue_size=1)
        
        rospy.loginfo("=== SAN SANG ===")

    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                self.latest_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
        except Exception as e:
            rospy.logerr_throttle(5, f"Lỗi đọc camera: {e}")

    def run(self):
        rospy.loginfo("Đợi 3s để hệ thống ổn định...")
        time.sleep(3)
        rospy.loginfo("=== BẮT ĐẦU ĐIỀU KHIỂN BẰNG AI ===")
        
        rate = rospy.Rate(20) # Vòng lặp 30 FPS
        
        while not rospy.is_shutdown():
            if self.latest_image is None:
                self.racer.stop()
                rate.sleep()
                continue
                
            # 1. TIỀN XỬ LÝ ẢNH
            frame = self.latest_image
            h, w = frame.shape[:2]
            
            # Cắt ảnh giống hệt lúc huấn luyện (từ Y=144->480, X=0->640 với ảnh gốc 1280x480)
            # Dùng công thức tỷ lệ để tương thích lỡ camera thay đổi độ phân giải
            roi_y1 = int(144 * h / 480)
            roi_y2 = h
            roi_w = min(w, 640) 
            
            roi = frame[roi_y1:roi_y2, 0:roi_w]
            
            # Resize, Grayscale và chuẩn hóa
            img = cv2.resize(roi, (self.IMG_WIDTH, self.IMG_HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            
            # Đóng gói thành Tensor (Batch=1, Channel=1, H, W)
            img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 2. ĐƯA VÀO MẠNG NEURAL (INFERENCE)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                steer_ai = outputs[0, 0].item()
                throttle_ai = outputs[0, 1].item()
                
            # 3. ĐIỀU KHIỂN XE
            # Giới hạn an toàn (Clamp)
            steer = max(-1.0, min(1.0, steer_ai))
            throttle = max(0.0, min(1.0, throttle_ai))
            
            self.racer.steer(steer, throttle)
            
            # In ra Terminal (mỗi 0.5s để đỡ trôi log)
            rospy.loginfo_throttle(0.5, f"[AI] Steer: {steer:+.3f} | Throttle: {throttle:.3f}")
            
            rate.sleep()
            
        self.racer.stop()
        rospy.loginfo("Kết thúc.")

def main():
    rospy.init_node('speed_track_ai_controller', anonymous=True)
    try:
        controller = SpeedTrackAIController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Lỗi: {e}")
    finally:
        try: RacerController().stop()
        except: pass

if __name__ == '__main__':
    main()
