import unittest
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.blackboard import Blackboard
from src.fsm.fsm_manager import FSMManager
from src.control.pid_controller import PIDController
from src.perception.camera.camera_processor import CameraProcessor
from src.perception.lidar.lidar_processor import LidarProcessor
import numpy as np
import cv2

class TestBlackboardFlow(unittest.TestCase):
    def setUp(self):
        self.blackboard = Blackboard()
        
        self.fsm = FSMManager()
        self.controller = PIDController(self.blackboard)
        self.camera = CameraProcessor(self.blackboard)
        self.lidar = LidarProcessor(self.blackboard)
        
        # Mocks hardware internally
        self.controller.initialize()
        
    def test_normal_driving_flow(self):
        # Giả lập dữ liệu Lidar (Không có vật cản gần)
        scan_data = [(0.0, 1.5), (30.0, 2.0), (-30.0, 2.0)]
        self.blackboard.set('latest_scan', scan_data)
        
        # Giả lập dữ liệu Camera (Một khung hình đen với vạch trắng ở giữa)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Vẽ 2 biên đường để CameraProcessor tính được center_x = 150
        cv2.line(img, (100, 0), (100, 300), (255, 255, 255), 5)
        cv2.line(img, (200, 0), (200, 300), (255, 255, 255), 5)
        self.blackboard.set('latest_image', img)
        
        # Chu kỳ chạy Blackboard (Tương tự trong vòng lặp ROS)
        self.lidar.process(self.blackboard)
        self.fsm.process(self.blackboard)
        self.camera.process(self.blackboard)
        self.controller.process(self.blackboard)
        
        # Kiểm tra kết quả
        self.assertEqual(self.blackboard.get('state_name'), 'NORMAL')
        self.assertAlmostEqual(self.blackboard.get('current_offset_px'), 0.0)
        
        center_x = self.blackboard.get('center_x')
        self.assertIsNotNone(center_x)
        # Tâm ảnh là 150, vạch cũng ở 150
        
        steering = self.blackboard.get('steering')
        self.assertIsNotNone(steering)
        # Xe đi thẳng thì steering phải xấp xỉ 0
        self.assertAlmostEqual(steering, 0.0, places=1)

    def test_dodging_flow(self):
        # Giả lập có vật cản gần phía trước (dưới TRIGGER_DIST) bên trái
        scan_data = [(10.0, 0.3)] # Góc dương (trái), khoảng cách 0.3m
        self.blackboard.set('latest_scan', scan_data)
        
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.line(img, (100, 0), (100, 300), (255, 255, 255), 5)
        cv2.line(img, (200, 0), (200, 300), (255, 255, 255), 5)
        self.blackboard.set('latest_image', img)
        
        self.lidar.process(self.blackboard)
        self.fsm.process(self.blackboard)
        
        # Vì có vật cản bên trái, xe phải né sang phải
        self.assertEqual(self.blackboard.get('state_name'), 'DODGING')
        self.assertEqual(self.blackboard.get('dodge_direction'), 1.0) # 1.0 là né phải

if __name__ == '__main__':
    unittest.main()
