import sys
import os
import time

# Thêm đường dẫn thư mục cha để có thể import src.core.control.racer_controller
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_city_modules.autonomous_modules import GoStraightModule, TurnModule
from src.core.control.racer_controller import RacerController

def test_go_straight(car):
    print("\n" + "="*50)
    print("--- 1. TEST GO STRAIGHT MODULE ---")
    # Khởi tạo module đi thẳng
    go_straight_ctrl = GoStraightModule(img_width=640, img_height=480, base_speed=0.3)
    
    # Kịch bản 1: Giả lập thấy decision node lệch bên phải (x = 420 so với tâm 320)
    # Và corner node nằm xa hơn (y = 150)
    mock_detections = [
        {"label": "decision", "x": 420, "y": 300},
        {"label": "corner", "x": 200, "y": 150} 
    ]
    
    print(">> Đầu vào (Detections):", mock_detections)
    speed, steering = go_straight_ctrl.calculate_command(mock_detections)
    
    if speed is not None:
        print(f">> Tính toán lệnh: Tốc độ = {speed}, Góc bẻ lái = {steering:.2f} (Dương -> Lệch phải)")
        print(">> Truyền lệnh xuống xe chạy trong 2 giây...")
        car.steer(steering, speed)
        time.sleep(2.0)
        car.stop()
        print(">> Xe đã dừng.")
    else:
        print(">> Không tìm thấy mục tiêu.")

def test_turn_interact(car):
    print("\n" + "="*50)
    print("--- 2. TEST TURN MODULE (INTERACT + BIỂN BÁO RẼ PHẢI) ---")
    # Khởi tạo module rẽ (giảm max_speed xuống 0.4 để test cho an toàn trên bàn/phòng thí nghiệm)
    turn_ctrl = TurnModule(img_width=640, turn_duration=2.0, max_speed=0.4, max_steering=1.0)
    
    # Kịch bản 2: Đến ngã tư có biển báo rẽ phải
    mock_detections = [
        {"label": "interact", "x": 320, "y": 400},
        {"label": "turn_right", "x": 500, "y": 100}
    ]
    print(">> Đầu vào (Detections):", mock_detections)
    
    if turn_ctrl.trigger_turn_if_needed(mock_detections):
        while True:
            speed, steering = turn_ctrl.process()
            if speed is None:  # Kết thúc thời gian rẽ (2.5s)
                break
            
            # Đưa lệnh xuống xe (do loop chạy rất nhanh, ta có thể dùng sleep một chút)
            car.steer(steering, speed)
            time.sleep(0.05)
        
        car.stop()
        print(">> Hoàn thành chu kỳ rẽ phải tự động và dừng xe.")
    else:
        print(">> Không thỏa mãn điều kiện rẽ.")

def test_turn_corner(car):
    print("\n" + "="*50)
    print("--- 3. TEST TURN MODULE (TỰ ĐỘNG ĐOÁN HƯỚNG BẰNG CORNER NODE) ---")
    turn_ctrl = TurnModule(img_width=640, turn_duration=2.5, max_speed=0.4, max_steering=1.0)
    
    # Kịch bản 3: Chỉ thấy góc cua (corner) nằm ở bên trái (x=100) -> Đường đang cong sang phải
    mock_detections = [
        {"label": "corner", "x": 100, "y": 300}
    ]
    print(">> Đầu vào (Detections):", mock_detections)
    
    if turn_ctrl.trigger_turn_if_needed(mock_detections):
        while True:
            speed, steering = turn_ctrl.process()
            if speed is None:
                break
            
            # Truyền lệnh rẽ
            car.steer(steering, speed)
            time.sleep(0.05)
            
        car.stop()
        print(">> Hoàn thành chu kỳ rẽ phải do tránh góc bên trái.")
    else:
        print(">> Không thỏa mãn điều kiện rẽ.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test các module điều khiển xe (Smart City).")
    parser.add_argument('--test', type=str, default='all', 
                        choices=['all', 'straight', 'interact', 'corner'],
                        help="Chọn bài test: all, straight, interact, corner")
    args = parser.parse_args()

    print("="*50)
    print("ĐANG KHỞI TẠO XE (RacerController)...")
    try:
        car = RacerController(config={"I2C_ADDRESS": 0x40})
        print("Khởi tạo thành công!")
    except Exception as e:
        print("Lỗi khởi tạo xe:", e)
        sys.exit(1)
        
    time.sleep(1) # Chờ phần cứng ổn định
    
    # Chạy các kịch bản test dựa trên tham số truyền vào
    if args.test in ['all', 'straight']:
        test_go_straight(car)
        time.sleep(1)
    
    if args.test in ['all', 'interact']:
        test_turn_interact(car)
        time.sleep(1)
    
    if args.test in ['all', 'corner']:
        test_turn_corner(car)
    
    print("\n" + "="*50)
    print(f"ĐÃ HOÀN TẤT CHẾ ĐỘ TEST: {args.test.upper()}.")
