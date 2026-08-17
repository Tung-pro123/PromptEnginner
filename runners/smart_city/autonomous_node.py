#!/usr/bin/env python3
import rospy
import cv2
import time
import os
import sys
import torch
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# from robot.perception.segmentation import SegLaneDetector
# from robot.perception.traffic_detector import TrafficSignDetector
# from robot.perception.lidar_processor import LidarProcessor
from robot.control.racer_controller import RacerController
from robot.ai.imitation_net import SmartCityImitationNet

class AutonomousDriver:
    def __init__(self):
        rospy.init_node('imitation_autonomous_driver', anonymous=True)
        
        self.bridge = CvBridge()
        self.robot = RacerController()
        
        # --- State Variables ---
        self.latest_image = None
        self.latest_scan = None
        
        # --- Load AI Model ---
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'imitation_model.pth')
        self.model = SmartCityImitationNet(input_dim=5, output_dim=2)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval() # Chuyển sang chế độ Inference (Tắt Dropout/BatchNorm)
            rospy.loginfo(f"Đã tải thành công mô hình từ: {model_path}")
        else:
            rospy.logerr(f"Không tìm thấy mô hình tại {model_path}. Vui lòng chạy file train_imitation.py trước!")
            sys.exit(1)
        
        # --- Subscribers ---
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        
        rospy.loginfo("Đã khởi tạo Node Chạy Tự Động. Bỏ tay cầm ra, xe sẽ tự chạy!")

    def camera_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def run(self):
        rate = rospy.Rate(30) # 30 Hz
        while not rospy.is_shutdown():
            if self.latest_image is not None and self.latest_scan is not None:
                # 1. Trích xuất State
                # lane_offset, lane_curvature = self.lane_detector.process(self.latest_image)
                # sign_class = self.sign_detector.process(self.latest_image)
                # dist_front, dist_side = self.lidar_processor.process(self.latest_scan)
                
                # Placeholder State
                lane_offset = 0.0
                lane_curvature = 0.0
                sign_class = 0
                dist_front = 10.0
                dist_side = 5.0
                
                # 2. Suy luận (Inference)
                with torch.no_grad():
                    # Đưa list về Tensor
                    state_tensor = torch.tensor([[lane_offset, lane_curvature, sign_class, dist_front, dist_side]], dtype=torch.float32)
                    
                    action = self.model(state_tensor) # Kích thước output: [1, 2]
                    
                    predicted_steer = action[0][0].item()
                    predicted_throttle = action[0][1].item()
                
                # 3. Điều khiển xe
                self.robot.steer(predicted_steer, predicted_throttle)

            rate.sleep()
            
        self.robot.stop()

if __name__ == '__main__':
    try:
        driver = AutonomousDriver()
        driver.run()
    except rospy.ROSInterruptException:
        pass
