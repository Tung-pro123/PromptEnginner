import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.base_controller import BaseController
from config import settings

class PController(BaseController):
    def __init__(self):
        self.kp = settings.Kp
        self.racer = None
        
    def initialize(self):
        try:
            # Tạm thời mock thư viện RacerController (Waveshare) nếu chưa có
            # from jetracer.nvidia_racecar import NvidiaRacecar
            # self.racer = NvidiaRacecar()
            print("[INFO] PController initialized. RacerController is simulated.")
        except Exception as e:
            print(f"[ERROR] Failed to init RacerController: {e}")

    def calculate_steering(self, center_x, current_offset_px):
        """Tính toán góc lái dựa trên thuật toán P Controller."""
        error_px = (center_x + current_offset_px) - settings.IMAGE_CENTER_X
        steering = error_px * self.kp
        
        # Giới hạn góc lái
        steering = max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
        return steering

    def move(self, speed, direction):
        if self.racer:
            self.racer.throttle = speed
            self.racer.steering = direction
        else:
            # In ra lệnh nếu chạy chế độ test
            pass # Lược bỏ print để tránh rác màn hình liên tục

    def stop(self):
        if self.racer:
            self.racer.throttle = 0.0
            self.racer.steering = 0.0
            print("[INFO] Robot stopped.")
