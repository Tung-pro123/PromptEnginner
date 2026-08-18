#!/usr/bin/env python3
import rospy
import time
import os
import sys
import csv
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sensor_msgs.msg import Image, LaserScan, Joy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from robot.control.racer_controller import RacerController
from robot.ai.imitation_net import SmartCityImitationNet, ImitationDataset

class InteractiveLearningNode:
    def __init__(self):
        rospy.init_node('interactive_learning_node', anonymous=True)
        self.robot = RacerController()
        
        # --- Config & Paths ---
        self.log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'imitation_model.pth')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # --- State Variables ---
        self.latest_image = None
        self.latest_scan = None
        
        # --- Joy / Control Variables ---
        self.joy_steer = 0.0
        self.joy_throttle = 0.0
        self.mode = "MANUAL" # Có 2 chế độ: MANUAL (Thu thập) và AUTONOMOUS (Test)
        self.is_training = False
        
        # --- CSV Writer cho lúc Thu thập (MANUAL) ---
        self.csv_file = None
        self.csv_writer = None
        self.start_new_log()
        
        # --- AI Model ---
        self.model = SmartCityImitationNet(input_dim=5, output_dim=2)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            rospy.loginfo("[+] Đã tải model cũ thành công!")
        self.model.eval()

        # --- Subscribers ---
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        rospy.loginfo("=== INTERACTIVE LEARNING NODE ===")
        rospy.loginfo("Nút A (Index 0): Chuyển đổi MANUAL <-> AUTONOMOUS")
        rospy.loginfo("Nút B (Index 1): Dừng xe và BẮT ĐẦU TRAIN dữ liệu ngay lập tức")

    def start_new_log(self):
        if self.csv_file:
            self.csv_file.close()
        ts = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.log_dir, f'dataset_{ts}.csv')
        self.csv_file = open(filepath, 'w', newline='')
        fieldnames = ['timestamp', 'lane_offset', 'lane_curvature', 'sign_class', 'lidar_dist_front', 'lidar_dist_side', 'action_steer', 'action_throttle']
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        rospy.loginfo(f"[!] Bắt đầu ghi log mới tại: {filepath}")

    def camera_callback(self, msg):
        try:
            # Chuyển đổi không cần cv_bridge
            if hasattr(msg, 'format') and 'jpeg' in msg.format.lower():
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if 'rgb' in msg.encoding: 
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.latest_image = img
        except Exception as e:
            rospy.logerr(f"Lỗi đọc ảnh: {e}")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def joy_callback(self, msg):
        self.joy_steer = msg.axes[0]
        self.joy_throttle = msg.axes[1]
        
        # Nút A (Index 0): Chuyển đổi MANUAL (Thu thập) và AUTONOMOUS (Tự lái)
        btn_a = msg.buttons[0]
        if btn_a and not hasattr(self, '_btn_a_pressed'):
            self._btn_a_pressed = True
            if self.mode == "MANUAL":
                self.mode = "AUTONOMOUS"
                rospy.logwarn(">>> CHUYỂN CHẾ ĐỘ: AUTONOMOUS (Tự lái) <<<")
            else:
                self.mode = "MANUAL"
                self.start_new_log() # Bắt đầu log mới khi con người lái lại
                rospy.logwarn(">>> CHUYỂN CHẾ ĐỘ: MANUAL (Thu thập dữ liệu) <<<")
        elif not btn_a:
            if hasattr(self, '_btn_a_pressed'):
                del self._btn_a_pressed
                
        # Nút B (Index 1): Bắt đầu Train ngay lập tức
        btn_b = msg.buttons[1]
        if btn_b and not self.is_training:
            self.is_training = True
            rospy.logerr(">>> DỪNG XE! BẮT ĐẦU HUẤN LUYỆN (TRAINING) TẠI CHỖ <<<")
            self.robot.stop()
            self.train_model()

    def train_model(self):
        # Đóng file csv hiện tại để chắc chắn dữ liệu đã lưu vào đĩa
        if self.csv_file:
            self.csv_file.flush()
            
        csv_files = glob.glob(os.path.join(self.log_dir, '*.csv'))
        if not csv_files:
            rospy.logerr("Không có file CSV nào để train!")
            self.is_training = False
            return
            
        dataset = ImitationDataset(csv_files)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        
        self.model.train()
        epochs = 20 # Số epoch nhỏ để train nhanh trực tiếp trên xe
        
        rospy.loginfo(f"Đang train {epochs} Epochs với {len(dataset)} mẫu...")
        for epoch in range(epochs):
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 5 == 0:
                rospy.loginfo(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
                
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()
        rospy.loginfo("[+] TRAIN XONG! Đã cập nhật Model mới. Có thể tiếp tục chạy.")
        self.is_training = False
        
        # Mở file CSV mới để tiếp tục log (nếu đang ở MANUAL)
        if self.mode == "MANUAL":
            self.start_new_log()

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.is_training:
                rate.sleep()
                continue
                
            if self.latest_image is not None and self.latest_scan is not None:
                # Placeholder State (thay thế bằng kết quả thật từ perception)
                lane_offset = 0.0
                lane_curvature = 0.0
                sign_class = 0
                dist_front = 10.0
                dist_side = 5.0
                
                # State Vector
                state_list = [lane_offset, lane_curvature, sign_class, dist_front, dist_side]
                
                if self.mode == "MANUAL":
                    # Lưu log
                    row = {
                        'timestamp': time.time(),
                        'lane_offset': f'{lane_offset:.4f}', 'lane_curvature': f'{lane_curvature:.4f}',
                        'sign_class': sign_class, 'lidar_dist_front': f'{dist_front:.2f}', 'lidar_dist_side': f'{dist_side:.2f}',
                        'action_steer': f'{self.joy_steer:.4f}', 'action_throttle': f'{self.joy_throttle:.4f}'
                    }
                    if self.csv_writer:
                        self.csv_writer.writerow(row)
                    
                    # Điều khiển xe bằng Joy
                    self.robot.steer(self.joy_steer, self.joy_throttle)
                    
                elif self.mode == "AUTONOMOUS":
                    # Suy luận bằng AI
                    with torch.no_grad():
                        state_tensor = torch.tensor([state_list], dtype=torch.float32)
                        action = self.model(state_tensor)
                        pred_steer = action[0][0].item()
                        pred_throttle = action[0][1].item()
                        
                    # Điều khiển xe bằng AI
                    self.robot.steer(pred_steer, pred_throttle)

            # Debug camera
            if self.latest_image is not None:
                cv2.imshow("Debug Camera", self.latest_image)
                cv2.waitKey(1)

            rate.sleep()
            
        cv2.destroyAllWindows()
        if self.csv_file:
            self.csv_file.close()
        self.robot.stop()

if __name__ == '__main__':
    try:
        node = InteractiveLearningNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
