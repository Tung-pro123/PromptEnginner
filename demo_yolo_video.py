import cv2
import os
import sys

def run_demo():
    # Kiểm tra xem có ultralytics chưa
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Lỗi: Không tìm thấy thư viện ultralytics. Hãy chạy: pip install ultralytics")
        sys.exit(1)

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'yolo.pt'))
    
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy mô hình tại {model_path}")
        sys.exit(1)

    print(f"Đang load mô hình từ {model_path}...")
    model = YOLO(model_path)

    # Chọn video đầu vào từ thư mục logs
    video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs', 'session_14', 'raw_camera.mp4'))
    
    if not os.path.exists(video_path):
        # Thử đường dẫn khác nếu không tìm thấy
        video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs', 'smart_city', 'raw_camera.avi'))
        if not os.path.exists(video_path):
            print(f"Lỗi: Không tìm thấy video demo tại {video_path}")
            sys.exit(1)

    print(f"Đang mở video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        sys.exit(1)
        
    # Lấy thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle nan or 0
        fps = 20.0
        
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'demo_output.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Bắt đầu xử lý video... Kết quả sẽ lưu tại: {output_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Chạy inference với confidence cao hơn (vd: 0.5)
        results = model.predict(frame, conf=0.5, verbose=False)
        r = results[0]
        
        # Lọc để mỗi class (decision, interact, corner, v.v.) chỉ giữ lại 1 box có confidence cao nhất
        if len(r.boxes) > 0:
            best_idx_per_class = {}
            for i in range(len(r.boxes)):
                cls_idx = int(r.boxes.cls[i].item())
                conf = r.boxes.conf[i].item()
                if cls_idx not in best_idx_per_class or conf > best_idx_per_class[cls_idx][1]:
                    best_idx_per_class[cls_idx] = (i, conf)
            
            keep_indices = [v[0] for v in best_idx_per_class.values()]
            r = r[keep_indices] # Giữ lại các box tốt nhất
            
        annotated_frame = r.plot()
        
        # Ghi frame đã được vẽ kết quả vào video đầu ra
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    print("Hoàn thành! Bạn có thể xem kết quả tại:", output_path)

if __name__ == "__main__":
    run_demo()
