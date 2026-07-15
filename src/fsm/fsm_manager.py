import time
import sys
import os

# Thêm đường dẫn src vào sys.path để import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class State:
    NORMAL = 0
    DODGING = 1
    REENTERING = 2

class FSMManager:
    def __init__(self):
        self.current_state = State.NORMAL
        self.dodge_direction = 0.0 # 1.0 (Right), -1.0 (Left)
        self.target_offset_px = 0.0
        self.current_offset_px = 0.0
        
        self.dodge_start_time = 0
        self.clear_frame_count = 0

    def update_from_lidar(self, front_dist, closest_angle, side_clear):
        """Cập nhật trạng thái máy dựa trên dữ liệu Lidar."""
        
        # 1. Từ NORMAL -> DODGING
        if self.current_state == State.NORMAL:
            if front_dist < settings.TRIGGER_DIST:
                self.current_state = State.DODGING
                self.dodge_start_time = time.time()
                self.clear_frame_count = 0
                
                # Quyết định hướng né
                if closest_angle < 0: # Vật cản bên phải -> Né trái
                    self.dodge_direction = -1.0
                    self.target_offset_px = -settings.DODGE_OFFSET_PX
                else: # Vật cản bên trái -> Né phải
                    self.dodge_direction = 1.0
                    self.target_offset_px = settings.DODGE_OFFSET_PX
                    
        # 2. Từ DODGING -> REENTERING
        elif self.current_state == State.DODGING:
            # Kiểm tra an toàn sườn xe
            if side_clear:
                self.clear_frame_count += 1
            else:
                self.clear_frame_count = 0
                
            watchdog_triggered = (time.time() - self.dodge_start_time) > settings.WATCHDOG_TIMEOUT
            
            if self.clear_frame_count >= settings.CLEAR_FRAMES_REQUIRED or watchdog_triggered:
                self.current_state = State.REENTERING
                self.target_offset_px = 0.0 # Về lại tâm đường
                
        # 3. Từ REENTERING -> NORMAL
        elif self.current_state == State.REENTERING:
            if abs(self.current_offset_px) < 1.0:
                self.current_state = State.NORMAL
                self.dodge_direction = 0.0

    def update_offset(self):
        """Cập nhật offset mượt mà (Ramping)."""
        diff = self.target_offset_px - self.current_offset_px
        if abs(diff) > settings.OFFSET_STEP:
            self.current_offset_px += settings.OFFSET_STEP if diff > 0 else -settings.OFFSET_STEP
        else:
            self.current_offset_px = self.target_offset_px
            
        return self.current_offset_px

    def get_state_name(self):
        if self.current_state == State.NORMAL: return "NORMAL"
        if self.current_state == State.DODGING: return "DODGING"
        if self.current_state == State.REENTERING: return "REENTERING"
        return "UNKNOWN"

    def process(self, blackboard):
        front_dist = blackboard.get('front_dist', 999.0)
        closest_angle = blackboard.get('closest_angle', 0.0)
        side_clear = blackboard.get('side_clear', True)
        
        self.update_from_lidar(front_dist, closest_angle, side_clear)
        current_offset_px = self.update_offset()
        
        blackboard.set('dodge_direction', self.dodge_direction)
        blackboard.set('current_offset_px', current_offset_px)
        blackboard.set('state_name', self.get_state_name())
