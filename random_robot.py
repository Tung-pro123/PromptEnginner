import time
import random

# Dưới đây là các hàm giả lập điều khiển động cơ.
# Bạn sẽ cần thay thế phần print bằng code điều khiển phần cứng thực tế
# (ví dụ: sử dụng Jetson.GPIO, gpiozero, hoặc thư viện của mạch L298N/PCA9685).

def move_forward(speed=0.5):
    print(f"🔼 Robot TIẾN về phía trước (tốc độ: {speed:.2f})")
    # TODO: Thêm code bật motor chạy tới

def move_backward(speed=0.5):
    print(f"🔽 Robot LÙI lại (tốc độ: {speed:.2f})")
    # TODO: Thêm code bật motor chạy lùi

def turn_left(speed=0.5):
    print(f"◀️ Robot RẼ TRÁI (tốc độ: {speed:.2f})")
    # TODO: Thêm code điều khiển bánh trái lùi, bánh phải tới

def turn_right(speed=0.5):
    print(f"▶️ Robot RẼ PHẢI (tốc độ: {speed:.2f})")
    # TODO: Thêm code điều khiển bánh trái tới, bánh phải lùi

def stop():
    print("⏹️ Robot DỪNG LẠI")
    # TODO: Thêm code ngắt nguồn motor

def run_random_robot():
    """Hàm chính điều khiển robot chạy ngẫu nhiên."""
    # Danh sách các hành động có thể thực hiện
    actions = [move_forward, move_backward, turn_left, turn_right]
    
    print("🤖 Bắt đầu chương trình chạy ngẫu nhiên. Nhấn Ctrl+C để dừng.")
    
    try:
        while True:
            # 1. Chọn một hành động ngẫu nhiên từ danh sách
            current_action = random.choice(actions)
            
            # 2. Sinh tốc độ ngẫu nhiên từ 0.3 đến 1.0 (30% đến 100% công suất)
            speed = random.uniform(0.3, 1.0)
            
            # 3. Sinh thời gian chạy ngẫu nhiên từ 0.5 đến 2.5 giây
            duration = random.uniform(0.5, 2.5)
            
            # Thực thi hành động
            current_action(speed)
            
            # Chờ robot chạy trong khoảng thời gian đã định
            time.sleep(duration)
            
            # Dừng robot lại một chút (0.2s) trước khi chuyển sang hành động mới
            # để bảo vệ động cơ khỏi việc đảo chiều đột ngột
            stop()
            time.sleep(0.2)

    except KeyboardInterrupt:
        # Xử lý khi người dùng nhấn Ctrl+C
        print("\n⚠️ Đã nhận lệnh dừng từ người dùng.")
    finally:
        # Luôn đảm bảo robot dừng lại khi thoát chương trình
        stop()
        print("✅ Chương trình kết thúc an toàn.")

if __name__ == "__main__":
    run_random_robot()
