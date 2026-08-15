#!/usr/bin/env python3
import time
import sys
import os

# Thêm đường dẫn để import RacerController
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.control.racer_controller import RacerController

def test_speed_mapping():
    ctrl = RacerController()
    print("=== Speed & Throttle Calibration ===")
    print("Mục tiêu: Tìm quãng đường đi được trong 2 giây ứng với từng mức ga.")
    print("Cách đo: Đặt xe trên vạch xuất phát, nhập mức ga. Dùng thước đo quãng đường thực tế.")
    
    while True:
        try:
            val = input("\nNhập mức throttle (ví dụ: 0.15, 0.20) hoặc 'q' để thoát: ")
            if val.lower() == 'q':
                break
            throttle = float(val)
            
            print(f"Chuẩn bị chạy với ga {throttle} trong 2 giây...")
            time.sleep(1)
            print("CHẠY!")
            
            ctrl.forward(throttle)
            time.sleep(2.0)
            ctrl.stop()
            
            print("ĐÃ DỪNG. Vui lòng lấy thước đo quãng đường (mét).")
            print(f"-> Vận tốc thực tế = Quãng đường / 2.0 (m/s)")
            
        except ValueError:
            print("Vui lòng nhập số hợp lệ.")
        except KeyboardInterrupt:
            ctrl.stop()
            break
            
def test_friction_limit():
    ctrl = RacerController()
    print("\n=== Friction (Mu) Calibration ===")
    print("Mục tiêu: Tìm giới hạn bám đường trước khi xe bị trượt ly tâm.")
    print("Cách đo: Xe sẽ chạy vòng tròn (đánh lái chết 0.8). Tăng dần ga.")
    print("Khi thấy bánh xe bắt đầu trượt (drift), ghi lại mức ga đó.")
    
    throttle = 0.15
    while True:
        try:
            val = input(f"\nNhấn Enter để chạy vòng tròn với ga {throttle}, hoặc 'q' để thoát: ")
            if val.lower() == 'q':
                break
            
            print(f"Đang chạy vòng tròn (ga {throttle}, steering 0.8)...")
            ctrl.steer(0.8, throttle)
            time.sleep(3.0)
            ctrl.stop()
            
            print("Xe có bị trượt không? Nếu không, hãy tăng ga cho lần tới.")
            throttle += 0.05
            
        except KeyboardInterrupt:
            ctrl.stop()
            break

if __name__ == '__main__':
    print("Chọn chế độ Calibration:")
    print("1. Đo tốc độ thẳng (Throttle -> m/s)")
    print("2. Đo độ bám đường (Tìm mu)")
    choice = input("Nhập 1 hoặc 2: ")
    if choice == '1':
        test_speed_mapping()
    elif choice == '2':
        test_friction_limit()
    else:
        print("Không hợp lệ.")
