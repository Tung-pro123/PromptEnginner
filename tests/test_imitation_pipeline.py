import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Thêm đường dẫn project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Giả lập (Mock) rospy và cv_bridge để có thể test mà không cần ROS
sys.modules['rospy'] = MagicMock()
sys.modules['cv_bridge'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['robot.control.racer_controller'] = MagicMock()

from runners.smart_city.data_collector_node import ImitationDataCollector
from runners.smart_city.autonomous_node import AutonomousDriver

class TestImitationPipeline(unittest.TestCase):
    
    @patch('runners.smart_city.data_collector_node.rospy')
    def test_data_collector_initialization(self, mock_rospy):
        """Kiểm tra Data Collector khởi tạo đúng và subscribe đủ topic."""
        collector = ImitationDataCollector()
        
        self.assertFalse(collector.is_recording)
        self.assertEqual(collector.current_steer, 0.0)
        self.assertEqual(collector.current_throttle, 0.0)
        
        # Kiểm tra xem có tạo log dir không
        self.assertTrue(os.path.exists(collector.log_dir))

    @patch('runners.smart_city.data_collector_node.rospy')
    def test_joy_callback(self, mock_rospy):
        """Kiểm tra xử lý tín hiệu tay cầm (Joy)."""
        collector = ImitationDataCollector()
        
        # Giả lập tin nhắn Joy
        mock_msg = MagicMock()
        mock_msg.axes = [0.5, 0.8]  # Steer = 0.5, Throttle = 0.8
        mock_msg.buttons = [0, 0, 0, 0] # Nút ghi (index 0) không được nhấn
        
        collector.joy_callback(mock_msg)
        
        self.assertEqual(collector.current_steer, 0.5)
        self.assertEqual(collector.current_throttle, 0.8)
        self.assertFalse(collector.is_recording)
        
        # Nhấn nút ghi (Toggle ON)
        mock_msg.buttons[0] = 1
        collector.joy_callback(mock_msg)
        self.assertTrue(collector.is_recording)
        self.assertIsNotNone(collector.csv_writer)
        
        # Nhả nút
        mock_msg.buttons[0] = 0
        collector.joy_callback(mock_msg)
        
        # Nhấn lại nút ghi (Toggle OFF)
        mock_msg.buttons[0] = 1
        collector.joy_callback(mock_msg)
        self.assertFalse(collector.is_recording)

    @patch('runners.smart_city.autonomous_node.rospy')
    def test_autonomous_driver_initialization(self, mock_rospy):
        """Kiểm tra Autonomous Driver khởi tạo."""
        driver = AutonomousDriver()
        self.assertIsNotNone(driver.robot)
        self.assertIsNone(driver.latest_image)
        self.assertIsNone(driver.latest_scan)

if __name__ == '__main__':
    unittest.main()
