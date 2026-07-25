#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

# Thêm root của project vào sys.path để giải quyết import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.blackboard import Blackboard
from src.perception.camera.camera_processor import CameraProcessor
from src.config import settings

def main():
    image_path = os.path.join(project_root, 'data', 'camera-noobstacle.jpg')
    output_path = os.path.join(project_root, 'data', 'camera-noobstacle-detected.jpg')

    print(f"[INFO] Nạp ảnh mẫu từ: {image_path}")
    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy ảnh tại: {image_path}")
        return

    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Không thể giải mã tệp ảnh.")
        return

    # Kích thước ảnh cấu hình (300x300)
    w, h = settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT
    resized_img = cv2.resize(img, (w, h))

    # 2. Khởi tạo Blackboard và đưa ảnh đầu vào lên đó
    blackboard = Blackboard()
    blackboard.set('latest_image', resized_img)
    blackboard.set('dodge_direction', 0.0) # Không né vật cản

    # 3. Khởi tạo CameraProcessor (Module xử lý chính trong src)
    processor = CameraProcessor(blackboard)
    
    # 4. Thực thi module xử lý (Cập nhật kết quả vào Blackboard)
    processor.process(blackboard)

    # 5. Lấy kết quả từ Blackboard theo đúng thiết kế nguyên lý
    center_x = blackboard.get('center_x')
    waypoints = blackboard.get('lane_waypoints')

    print(f"\n[INFO] Đã thực thi module CameraProcessor qua Blackboard thành công!")
    print(f"--- THÔNG SỐ ĐỌC TỪ BLACKBOARD ---")
    print(f" * Tâm làn đường (center_x): {center_x}")
    print(f" * Điểm mốc quỹ đạo (lane_waypoints): {waypoints}")

    # 6. Vẽ ảnh minh họa (Overlay & Threshold) để kiểm tra độ chính xác
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, settings.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    overlay = resized_img.copy()

    # Vẽ đường ngang scanlines
    y_lines = [160, 200, 240, 280]
    for y in y_lines:
        cv2.line(overlay, (0, y), (w, y), (0, 255, 255), 1)

    # Vẽ các waypoint từ Blackboard
    for i, pt in enumerate(waypoints):
        cv2.circle(overlay, pt, 5, (0, 0, 255), -1)
        cv2.putText(overlay, f"P{i}", (pt[0] + 8, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Nối các waypoint thành đường quỹ đạo
    if len(waypoints) > 1:
        for i in range(1, len(waypoints)):
            cv2.line(overlay, waypoints[i-1], waypoints[i], (0, 255, 0), 2)

    # Vẽ tâm xe màu xanh dương
    cv2.line(overlay, (settings.IMAGE_CENTER_X, 0), (settings.IMAGE_CENTER_X, h), (255, 0, 0), 1)
    
    # Tính độ lệch lái và vẽ mũi tên chỉ hướng
    deviation = center_x - settings.IMAGE_CENTER_X
    cv2.arrowedLine(overlay, (settings.IMAGE_CENTER_X, 260), (int(center_x), 260), (255, 100, 0), 2, tipLength=0.3)
    cv2.putText(overlay, f"Offset: {deviation:.1f}px", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Ghép so sánh 2 ảnh
    combined = np.hstack((overlay, thresh_bgr))

    # Lưu kết quả kiểm tra
    cv2.imwrite(output_path, combined)
    print(f"[SUCCESS] Đã lưu hình ảnh kiểm thử vào: {output_path}")

if __name__ == '__main__':
    main()
