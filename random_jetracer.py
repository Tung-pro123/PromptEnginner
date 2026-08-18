import time
import random
import sys

# Import thư viện chuẩn của JetRacer
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    car = NvidiaRacecar()
except ImportError:
    print("❌ Không tìm thấy thư viện jetracer!")
    print("Vui lòng đảm bảo bạn đang chạy code này trên xe JetRacer và đã cài đặt thư viện đầy đủ.")
    sys.exit(1)

def stop():
    """Dừng hoàn toàn xe."""
    car.throttle = 0.0
    car.steering = 0.0
    print("⏹️ Xe DỪNG")

def run_random_jetracer():
    """Điều khiển JetRacer chạy ngẫu nhiên."""
    print("🏎️ Bắt đầu chương trình chạy ngẫu nhiên cho JetRacer. Nhấn Ctrl+C để dừng.")
    
    # JetRacer có cơ chế lái giống ô tô thật (Ackermann steering): 
    # Cần 2 thông số: bướm ga (throttle) và góc lái (steering)
    # Cấu trúc: (hệ số bướm ga, góc lái)
    # - Góc lái: -1.0 (hết lái trái) -> 1.0 (hết lái phải)
    # - Bướm ga: -1.0 (lùi tối đa) -> 1.0 (tiến tối đa)
    actions = [
        (1.0, 0.0),    # Tiến thẳng
        (-1.0, 0.0),   # Lùi thẳng
        (1.0, -1.0),   # Vừa tiến vừa rẽ trái
        (1.0, 1.0),    # Vừa tiến vừa rẽ phải
        (-1.0, -1.0),  # Vừa lùi vừa rẽ trái
        (-1.0, 1.0),   # Vừa lùi vừa rẽ phải
    ]
    
    try:
        while True:
            # 1. Chọn một hướng chạy ngẫu nhiên
            action = random.choice(actions)
            
            # 2. Sinh tốc độ ngẫu nhiên.
            # LƯU Ý AN TOÀN: Động cơ JetRacer rất mạnh, chạy ngẫu nhiên dễ va chạm.
            # Tui giới hạn ga từ 0.15 đến 0.3 (15% đến 30% công suất) để an toàn.
            speed_base = random.uniform(0.15, 0.3)
            
            throttle = action[0] * speed_base
            steering = action[1]
            
            # 3. Sinh thời gian chạy từ 0.5 đến 1.5 giây
            duration = random.uniform(0.5, 1.5)
            
            print(f"⚙️ Lệnh mới - Ga: {throttle:.2f} | Lái: {steering:.2f} | Thời gian: {duration:.2f}s")
            
            # Truyền lệnh điều khiển xuống bánh xe!
            car.throttle = throttle
            car.steering = steering
            
            # Chờ xe chạy
            time.sleep(duration)
            
            # Dừng lại một nhịp 0.3s trước khi bẻ lái sang hướng mới
            stop()
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n⚠️ Đã nhận lệnh dừng khẩn cấp từ bạn!")
    finally:
        # Quan trọng nhất: luôn set throttle = 0 khi thoát để xe không đâm vào tường
        stop()
        print("✅ Đã ngắt động cơ JetRacer an toàn.")

if __name__ == "__main__":
    run_random_jetracer()
