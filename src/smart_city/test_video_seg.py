import os
import sys
import cv2

# Đảm bảo đường dẫn tới thư mục src gốc để import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.smart_city.config import SmartCityConfig
from src.smart_city.perception.seg_lane_detector import YoloSegDetector
from src.smart_city.perception.traffic_detector import TrafficDetector

def main():
    video_path = os.path.join(os.path.dirname(__file__), 'smart-city.mp4')
    if not os.path.exists(video_path):
        print(f"Không tìm thấy video tại: {video_path}")
        return

    print("=== Khởi tạo YoloSegDetector & TrafficDetector ===")
    cfg = SmartCityConfig()
    # Dùng file .pt để test trên PC tránh lỗi môi trường
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best.pt')
    
    detector = YoloSegDetector(model_path, cfg)
    sign_detector = TrafficDetector(cfg.image_width, cfg.image_height)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'smart-city-output.mp4')
    
    cap = cv2.VideoCapture(video_path)
    
    # Lấy thông số video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec chuẩn cho MP4
    out = cv2.VideoWriter(out_path, fourcc, fps, (cfg.image_width, cfg.image_height))
    
    print(f"Bắt đầu xử lý video... Kết quả sẽ lưu tại: {out_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize ảnh về kích thước chuẩn của Jetbot theo file config
        frame = cv2.resize(frame, (cfg.image_width, cfg.image_height))
        
        # Chạy thuật toán nhận diện
        res = detector.detect(frame)
        
        # Vẽ giao diện Debug với Mask và điểm Setpoint
        debug_frame = detector.draw_debug(frame, res)
        
        # Thêm text thông tin của YOLO-seg
        cv2.putText(debug_frame, f"Mode: {res.mode} | Setpoint: {res.center_x}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        # Chạy thêm thuật toán nhận diện biển báo (TrafficDetector) giống hệt trên xe thật
        light, sign = sign_detector.detect(frame)
        
        # Hiển thị biển báo lên góc trên bên phải màn hình
        cv2.putText(debug_frame, f"SIGN: {sign}", (cfg.image_width - 150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if sign != "NONE" else (200, 200, 200), 2)
                    
        # Mô phỏng quyết định lái (Lái giả lập)
        decision = "STRAIGHT"
        if res.has_crosswalk:
            decision = f"INTERSECTION -> {sign}"
            
        cv2.putText(debug_frame, f"ACTION: {decision}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(debug_frame)
        
        # Hiển thị trực tiếp lên màn hình
        cv2.imshow("JetRacer YOLO-seg Live Test", debug_frame)
        
        # Nhấn phím 'q' để thoát sớm
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đã ấn phím Q để dừng test video sớm.")
            break
            
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("=== Hoàn tất xử lý video! ===")

if __name__ == '__main__':
    main()
