import unittest
import sys
import os

# Đảm bảo có thể import từ src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fsm.fsm_manager import FSMManager, State
from src.config import settings

class TestFSMManager(unittest.TestCase):
    def setUp(self):
        """Hàm này chạy trước mỗi test case."""
        self.fsm = FSMManager()
        # Đặt lại hằng số watchdog thấp cho dễ test (tuỳ chọn)
        settings.WATCHDOG_TIMEOUT = 1.0

    def test_initial_state(self):
        """Kiểm tra trạng thái ban đầu luôn là NORMAL."""
        self.assertEqual(self.fsm.current_state, State.NORMAL)
        self.assertEqual(self.fsm.get_state_name(), "NORMAL")

    def test_trigger_dodging_right(self):
        """Kiểm tra chuyển từ NORMAL sang DODGING khi có vật cản bên trái (né phải)."""
        # Giả lập khoảng cách < 0.70m, góc dương (bên trái)
        self.fsm.update_from_lidar(front_dist=0.5, closest_angle=10.0, side_clear=True)
        
        self.assertEqual(self.fsm.current_state, State.DODGING)
        self.assertEqual(self.fsm.dodge_direction, 1.0) # Né phải
        self.assertGreater(self.fsm.target_offset_px, 0) # Offset phải dương

    def test_trigger_dodging_left(self):
        """Kiểm tra chuyển từ NORMAL sang DODGING khi có vật cản bên phải (né trái)."""
        # Giả lập khoảng cách < 0.70m, góc âm (bên phải)
        self.fsm.update_from_lidar(front_dist=0.6, closest_angle=-15.0, side_clear=True)
        
        self.assertEqual(self.fsm.current_state, State.DODGING)
        self.assertEqual(self.fsm.dodge_direction, -1.0) # Né trái
        self.assertLess(self.fsm.target_offset_px, 0) # Offset phải âm
        
    def test_reentering_after_clear(self):
        """Kiểm tra xe có quay lại làn (REENTERING) khi sườn xe an toàn không."""
        self.fsm.current_state = State.DODGING
        settings.CLEAR_FRAMES_REQUIRED = 3
        
        # Mô phỏng sườn an toàn trong 3 frame liên tiếp
        self.fsm.update_from_lidar(0.8, 0, side_clear=True)
        self.fsm.update_from_lidar(0.8, 0, side_clear=True)
        self.fsm.update_from_lidar(0.8, 0, side_clear=True)
        
        self.assertEqual(self.fsm.current_state, State.REENTERING)
        self.assertEqual(self.fsm.target_offset_px, 0.0)

    def test_ramping_offset(self):
        """Kiểm tra tính năng tăng giảm offset mượt mà (S-Curve Ramp)."""
        self.fsm.current_offset_px = 0.0
        self.fsm.target_offset_px = 20.0
        settings.OFFSET_STEP = 5.0
        
        self.fsm.update_offset()
        self.assertEqual(self.fsm.current_offset_px, 5.0)
        
        self.fsm.update_offset()
        self.assertEqual(self.fsm.current_offset_px, 10.0)

if __name__ == '__main__':
    unittest.main()
