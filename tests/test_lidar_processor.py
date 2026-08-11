import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.perception.lidar.lidar_processor import LidarProcessor
from src.config import settings

class TestLidarProcessor(unittest.TestCase):
    def setUp(self):
        self.lidar = LidarProcessor()

    def test_empty_scan(self):
        """Trường hợp không có dữ liệu LiDAR."""
        front_dist, closest_angle, side_clear = self.lidar.process_scan([])
        self.assertEqual(front_dist, 999.0)
        self.assertEqual(closest_angle, 0.0)
        self.assertTrue(side_clear)

    def test_front_obstacle(self):
        """Kiểm tra tìm vật cản trước mặt."""
        # Giả lập vật cản ở 0 độ, cách 0.5m
        scan_data = [
            (0.0, 0.5),   # Trong FRONT_ANGLE_RANGE
            (45.0, 0.2),  # Ngoài FRONT_ANGLE_RANGE
            (-10.0, 0.6)  # Trong FRONT_ANGLE_RANGE nhưng xa hơn
        ]
        
        settings.FRONT_ANGLE_RANGE = 35.0
        front_dist, closest_angle, side_clear = self.lidar.process_scan(scan_data)
        
        self.assertEqual(front_dist, 0.5)
        self.assertEqual(closest_angle, 0.0)
        self.assertTrue(side_clear)

    def test_side_clear_detection(self):
        """Kiểm tra logic nhận diện an toàn sườn xe."""
        # Giả lập vật cản ở sườn xe (góc > 110 hoặc < -110)
        scan_data = [
            (120.0, 0.2), # Không an toàn (khoảng cách 0.2 < 0.3m)
        ]
        settings.SIDE_ANGLE_CLEAR = 110.0
        settings.SIDE_CLEAR_DIST = 0.3
        
        _, _, side_clear = self.lidar.process_scan(scan_data)
        self.assertFalse(side_clear)

    def test_noise_filtering(self):
        """Đảm bảo các giá trị nhiễu (distance < 0.05) bị bỏ qua."""
        scan_data = [
            (0.0, 0.01), # Nhiễu Lidar
            (5.0, 0.6)   # Thực tế
        ]
        front_dist, closest_angle, _ = self.lidar.process_scan(scan_data)
        
        self.assertEqual(front_dist, 0.6)
        self.assertEqual(closest_angle, 5.0)

if __name__ == '__main__':
    unittest.main()
