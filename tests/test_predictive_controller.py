#!/usr/bin/env python3
import sys
import os
import time

# Thêm thư mục gốc vào sys.path để import được src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.control.predictive_controller import PredictiveController
from robot.utils.blackboard import Blackboard

def debug_predictive_controller():
    print("="*50)
    print("    DEBUG PREDICTIVE CONTROLLER (HARDWARE & LOGIC) ")
    print("="*50)
    
    # 1. Khởi tạo Blackboard giả
    bb = Blackboard()
    
    # 2. Khởi tạo Controller
    print("[1] Đang khởi tạo kết nối phần cứng...")
    controller = PredictiveController(bb)
    controller.initialize()
    time.sleep(1)
    
    # 3. Test phần cứng: Lái (Steering)
    print("\n[2] Kiểm tra hệ thống lái (Servo):")
    print(" -> Bẻ lái hết mức sang TRÁI (-1.0)")
    controller.move(0.0, -1.0)
    time.sleep(1.5)
    
    print(" -> Bẻ lái hết mức sang PHẢI (1.0)")
    controller.move(0.0, 1.0)
    time.sleep(1.5)
    
    print(" -> Trả lái về GIỮA (0.0)")
    controller.move(0.0, 0.0)
    time.sleep(1.5)

    # 4. Test phần cứng: Ga (Throttle)
    print("\n[3] Kiểm tra hệ thống ga (Motor):")
    print(" -> Chạy TỚI tốc độ nhẹ (0.2)...")
    controller.move(0.2, 0.0)
    time.sleep(1.5)
    
    print(" -> Dừng xe!")
    controller.stop()
    time.sleep(1)

    print(" -> Chạy LÙI tốc độ nhẹ (-0.2)...")
    controller.move(-0.2, 0.0)
    time.sleep(1.5)
    
    print(" -> Dừng xe!")
    controller.stop()

    # 5. Test Thuật toán tính toán (Polyfit)
    print("\n[4] Kiểm tra thuật toán dự đoán (Predictive Polyfit):")
    # Giả lập truyền vào một mảng điểm waypoint (x, y)
    fake_waypoints = [(100, 300), (120, 260), (140, 220), (170, 180)]
    print(f" -> Cấp dữ liệu giả lập (Waypoints): {fake_waypoints}")
    
    bb.set('lane_waypoints', fake_waypoints)
    controller.process(bb)
    
    steer = bb.get('steering')
    curve = bb.get('predicted_curve')
    
    print(f" -> Kết quả góc lái xuất ra: {steer:.3f}")
    print(f" -> Các điểm trên đường cong dự đoán (Curve points):")
    for pt in curve:
        print(f"    - {pt}")

    # 6. Dọn dẹp
    controller.stop()
    print("\n" + "="*50)
    print("                HOÀN TẤT DEBUG!                 ")
    print("="*50)

if __name__ == '__main__':
    try:
        debug_predictive_controller()
    except KeyboardInterrupt:
        print("\n[!] Nhận lệnh ngắt (Ctrl+C). Đang dừng xe khẩn cấp...")
        try:
            r = PredictiveController(Blackboard())
            r.initialize()
            r.stop()
        except:
            pass
        sys.exit(0)
