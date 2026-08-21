import cv2
import os
import sys
import numpy as np

# Thêm đường dẫn để import từ thư mục cha/khác
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from smart_city_modules.autonomous_modules import DecisionModule, TurnModule

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
    
    decision_module = DecisionModule(img_width=width, img_height=height)
    turn_ctrl = TurnModule(img_width=width, turn_duration=2.0, max_speed=0.4, max_steering=1.0)
    
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
        
        # Trích xuất detections để đẩy vào DecisionModule
        detections = []
        if len(r.boxes) > 0:
            if r.masks is not None:
                for box, mask, cls in zip(r.boxes, r.masks, r.boxes.cls):
                    label = model.names[int(cls)]
                    xy = mask.xy[0]
                    if len(xy) > 0:
                        x = np.mean(xy[:, 0])
                        y = np.mean(xy[:, 1])
                        detections.append({"label": label, "x": float(x), "y": float(y)})
            else:
                for box, cls in zip(r.boxes, r.boxes.cls):
                    label = model.names[int(cls)]
                    x, y, w, h = box.xywh[0]
                    detections.append({"label": label, "x": float(x), "y": float(y)})
        
        # Lấy quyết định từ DecisionModule
        action, target_node, raw_speed, raw_steering, steps = decision_module.make_decision(detections)
        
        # Áp dụng logic của Autonomous Module (State Machine cho việc rẽ)
        if action in ["turn_left", "turn_right"] and not turn_ctrl.is_turning:
            turn_ctrl.start_turn("left" if action == "turn_left" else "right")
            
        turn_speed, turn_steering = turn_ctrl.process()
        
        if turn_speed is not None and turn_steering is not None:
            final_speed = turn_speed
            final_steering = turn_steering
            car_status = f"TURNING {turn_ctrl.current_direction.upper()}"
        else:
            final_speed = raw_speed
            final_steering = raw_steering
            car_status = "GOING STRAIGHT"
        
        # Vẽ Text trạng thái chính lên góc trái trên
        status_text = f"State: {car_status} | Speed: {final_speed:.2f} | Steer: {final_steering:.2f}"
        if target_node and car_status == "GOING STRAIGHT":
            status_text += f" | Target: {target_node['label']}"
            # Vẽ đường nối từ xe đến target
            cv2.line(annotated_frame, (int(width/2), height), (int(target_node['x']), int(target_node['y'])), (0, 255, 0), 2)
            
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # In các bước quyết định ra màn hình console và vẽ lên video
        y_offset = 60
        for step in steps:
            cv2.putText(annotated_frame, f"- {step}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            y_offset += 20
            
        if frame_count % 15 == 0:
            print(f"--- Frame {frame_count} ---")
            for step in steps:
                print("  >", step)
            print("  =>", status_text)
        
        # Ghi frame đã được vẽ kết quả vào video đầu ra
        out.write(annotated_frame)
        
        # Hiển thị ra màn hình cho dễ đánh giá cảm quan
        cv2.imshow("Smart City Robot Demo", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đã dừng phát video.")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames, last action: {action}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Hoàn thành! Bạn có thể xem kết quả tại:", output_path)

if __name__ == "__main__":
    run_demo()
