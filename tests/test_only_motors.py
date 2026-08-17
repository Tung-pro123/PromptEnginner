#!/usr/bin/env python3
"""
Kiểm tra RIÊNG LẺ: Động cơ và Servo lái (Ackermann Steering)
Chạy trực tiếp trên Jetson:
    python3 tests/test_only_motors.py
"""
import sys
# Sắp xếp lại sys.path để ưu tiên các thư viện Python 3 trước, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import os
import time

# Thêm thư mục gốc chứa src vào path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.control.racer_controller import RacerController

def main():
    print("="*60)
    print("🏎️  BẮT ĐẦU TEST RIÊNG BIỆT ĐỘNG CƠ & LÁI (JETRACER) 🏎️")
    print("="*60)
    print("LƯU Ý: HÃY ĐẶT XE KÊ CAO BÁNH LÊN KHỎI MẶT ĐẤT ĐỂ ĐẢM BẢO AN TOÀN!")
    input("Nhấn ENTER để bắt đầu test...")

    # Khởi tạo RacerController để kết nối phần cứng NvidiaRacecar
    controller = RacerController()
    
    try:
        # 1. Test Servo Lái (Steering)
        print("\n1. Test Servo Lái (Bánh trước):")
        
        print("  - Lái hết cỡ sang TRÁI (steering = -1.0)")
        controller.steer(-1.0, 0.0)  # Lái trái, ga = 0
        time.sleep(1.5)
        
        print("  - Lái hết cỡ sang PHẢI (steering = 1.0)")
        controller.steer(1.0, 0.0)   # Lái phải, ga = 0
        time.sleep(1.5)
        
        print("  - Trả lái về THẲNG (steering = 0.0)")
        controller.steer(0.0, 0.0)
        time.sleep(1.0)
        
        # 2. Test Động cơ Ga (Throttle)
        print("\n2. Test Động cơ kéo (Bánh sau):")
        
        # Lấy tốc độ chạy test an toàn (nhẹ)
        test_speed = 0.15
        
        print(f"  - Chạy TIẾN chậm (throttle = {test_speed}) trong 1.5 giây")
        controller.forward(test_speed)
        time.sleep(1.5)
        
        print("  - Dừng động cơ")
        controller.stop()
        time.sleep(1.0)
        
        print(f"  - Chạy LÙI chậm (throttle = -{test_speed}) trong 1.5 giây")
        controller._set_steering(0.0)
        controller._set_throttle(-test_speed)
        time.sleep(1.5)
        
        print("  - Dừng động cơ")
        controller.stop()
        
        print("\n" + "="*60)
        print("✅ ĐÃ HOÀN THÀNH TEST ĐỘNG CƠ AN TOÀN.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Bị ngắt bởi người dùng. Đảm bảo dừng xe!")
        controller.stop()
    except Exception as e:
        print(f"\n[ERROR] Lỗi điều khiển động cơ: {e}")
        controller.stop()

if __name__ == '__main__':
    main()
