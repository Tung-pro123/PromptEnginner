#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

# Thêm project root vào sys.path để import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.blackboard import Blackboard
from src.perception.camera.camera_processor import CameraProcessor
from src.config import settings

def main():
    # 1. Tìm đường dẫn video đầu vào
    possible_dirs = [
        os.path.join(project_root, 'logs', 'speed_track_no_obstacle'),
        os.path.join(project_root, 'logs', 'no_speed_track'),
    ]
    
    video_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            video_dir = d
            break
            
    if not video_dir:
        print(f"[ERROR] Không tìm thấy thư mục log chứa video tại bất kỳ đường dẫn nào sau đây:")
        for d in possible_dirs:
            print(f" - {d}")
        return

    video_path = os.path.join(video_dir, 'raw_camera.avi')
    output_path = os.path.join(video_dir, 'raw_camera_processed.avi')

    print(f"[INFO] Nạp video thô từ: {video_path}")
    if not os.path.exists(video_path):
        print(f"[ERROR] Không tìm thấy tệp video raw_camera.avi tại: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở video: {video_path}")
        return

    # Lấy thông số video đầu vào
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Định nghĩa kích thước ảnh xử lý từ settings
    w, h = settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT
    print(f"[INFO] Cấu hình xử lý: Kích thước = {w}x{h}, FPS = {fps}, Số khung hình = {frame_count}")

    # Khởi tạo VideoWriter cho đầu ra kết quả (3 ảnh ngang: overlay | BEV | threshold)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 3, h))

    # Khởi tạo blackboard, processor và controller để mô phỏng tính toán lệnh lái
    blackboard = Blackboard()
    processor = CameraProcessor(blackboard)
    
    from src.control.predictive_controller import PredictiveController
    from src.control.pid_controller import PIDController
    if settings.CONTROLLER_TYPE == 'predictive':
        controller = PredictiveController(blackboard)
    else:
        controller = PIDController(blackboard)
    controller.initialize()

    print("[INFO] Bắt đầu phân tích khung hình video...")
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize ảnh về kích thước xử lý của thuật toán
        resized_frame = cv2.resize(frame, (w, h))

        # Đưa ảnh lên blackboard
        blackboard.set('latest_image', resized_frame)
        blackboard.set('dodge_direction', 0.0)

        # Chạy CameraProcessor
        processor.process(blackboard)
        
        # Chạy Controller để tính toán các thông số điều khiển thực tế
        controller.process(blackboard)

        # Lấy kết quả phân tích
        center_x = blackboard.get('center_x', settings.IMAGE_CENTER_X)
        waypoints = blackboard.get('lane_waypoints', [])
        steering = blackboard.get('steering', 0.0)
        throttle = blackboard.get('throttle', 0.0)

        # Tạo ảnh hiển thị kết quả (Vẽ các waypoint bám làn)
        # Nếu USE_BOUNDARY_PATH, debug_img đã được vẽ sẵn waypoints trong detector
        # → Dùng ngay làm overlay; nếu không thì dùng frame gốc
        processed_image = blackboard.get('latest_image', None)
        if processed_image is not None and processed_image.shape == resized_frame.shape:
            overlay = processed_image.copy()
        else:
            overlay = resized_frame.copy()
            # Vẽ các waypoint phát hiện được (chỉ khi chưa được vẽ bởi detector)
            for i, pt in enumerate(waypoints):
                cv2.circle(overlay, pt, 4, (0, 0, 255), -1)
            # Vẽ đường nối các waypoint tạo thành đường chạy
            if len(waypoints) > 1:
                for i in range(1, len(waypoints)):
                    cv2.line(overlay, waypoints[i-1], waypoints[i], (0, 255, 0), 2)


        # Vẽ trục tâm xe (xanh dương) và mũi tên hướng bẻ lái thực tế
        cv2.line(overlay, (settings.IMAGE_CENTER_X, 0), (settings.IMAGE_CENTER_X, h), (255, 0, 0), 1)
        deviation = center_x - settings.IMAGE_CENTER_X
        cv2.arrowedLine(overlay, (settings.IMAGE_CENTER_X, h - 40), (int(center_x), h - 40), (255, 100, 0), 2, tipLength=0.3)
        
        # Xác định hướng bẻ lái thực tế
        if steering > 0.05:
            direction_str = "RIGHT"
            dir_color = (0, 0, 255)  # Đỏ cho rẽ phải
        elif steering < -0.05:
            direction_str = "LEFT"
            dir_color = (0, 255, 0)  # Xanh lá cho rẽ trái
        else:
            direction_str = "STRAIGHT"
            dir_color = (255, 255, 255)  # Trắng cho đi thẳng

        # Hiển thị các thông số điều khiển (Steer, Throttle, Direction, Deviation) lên màn hình
        cv2.putText(overlay, f"Steer: {steering:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(overlay, f"Throttle: {throttle:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(overlay, f"Dir: {direction_str}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dir_color, 1)
        cv2.putText(overlay, f"Dev: {deviation:.1f}px", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Lấy ảnh Bird's Eye View debug nếu có (chỉ tồn tại khi dùng USE_BOUNDARY_PATH)
        bev_debug = blackboard.get('bev_debug_img', None)
        camera_thresh = blackboard.get('camera_thresh', None)

        # Tạo ảnh panel thứ 2: BEV debug hoặc fallback sang threshold
        if bev_debug is not None and bev_debug.shape == (h, w, 3):
            panel2 = bev_debug.copy()
            cv2.putText(panel2, "BEV + Boundary Fit", (5, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        elif camera_thresh is not None:
            if len(camera_thresh.shape) == 2:
                panel2 = cv2.cvtColor(camera_thresh, cv2.COLOR_GRAY2BGR)
            else:
                panel2 = camera_thresh.copy()
            panel2 = cv2.resize(panel2, (w, h))
        else:
            panel2 = np.zeros((h, w, 3), dtype=np.uint8)

        # Tạo ảnh threshold nhị phân của camera để so sánh (panel 3)
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, settings.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.putText(thresh_bgr, "Grayscale Thresh", (5, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Ghép ngang 3 panel: Camera overlay | BEV debug | Threshold
        combined = np.hstack((overlay, panel2, thresh_bgr))

        # Ghi khung hình vào video đầu ra
        out.write(combined)

        processed_count += 1
        if processed_count % 50 == 0:
            print(f" -> Đã xử lý {processed_count}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"\n[SUCCESS] Phân tích hoàn tất! Video kết quả được lưu tại: {output_path}")

if __name__ == '__main__':
    main()
