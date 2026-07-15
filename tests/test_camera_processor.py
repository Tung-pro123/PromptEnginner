import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.perception.camera.camera_processor import CameraProcessor
from src.config import settings

class TestCameraProcessor(unittest.TestCase):
    def setUp(self):
        self.camera = CameraProcessor()
        settings.IMAGE_WIDTH = 300
        settings.IMAGE_HEIGHT = 300
        settings.IMAGE_CENTER_X = 150
        settings.THRESHOLD_VALUE = 180

    def test_process_empty_frame(self):
        """Nếu truyền vào None thì phải trả về tâm ảnh."""
        center_x = self.camera.process_frame(None)
        self.assertEqual(center_x, 150)

    def test_process_black_frame(self):
        """Nếu ảnh đen hoàn toàn (mất 2 biên), phải rà hướng đánh lái gần nhất."""
        black_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        self.camera.last_known_direction = 1.0 # Đang lệch phải
        
        center_x = self.camera.process_frame(black_frame)
        # Bẻ ngược lại nhẹ: center = 150 + (20 * 1) = 170
        self.assertEqual(center_x, 170)

    def test_process_single_line_dodging(self):
        """Mô phỏng trường hợp đang né (DODGING) và chỉ thấy 1 biên vạch."""
        # Tạo ảnh đen có 1 vạch trắng ở cột x=50
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[:, 45:55, :] = 255 
        
        self.camera.estimated_lane_width = 100.0
        
        # Đang né trái (dodge_direction = -1.0), vạch x=50 phải được hiểu là vạch biên TRÁI
        center_x = self.camera.process_frame(frame, dodge_direction=-1.0)
        # Tâm sẽ nằm bên phải vạch này: 50 + (100 / 2) = 100
        self.assertEqual(center_x, 100.0)

if __name__ == '__main__':
    unittest.main()
