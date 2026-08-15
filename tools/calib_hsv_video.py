import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def nothing(x):
    pass

def run_calibrator():
    # 1. Chọn file video log
    root = tk.Tk()
    root.withdraw() # Ẩn cửa sổ chính
    print("Hãy chọn file video .avi (VD: speed_racing_v2_...avi) để lấy mẫu màu mặt đường...")
    video_path = filedialog.askopenfilename(
        title="Chọn file Video Log",
        filetypes=[("AVI Video", "*.avi"), ("All Files", "*.*")]
    )
    
    if not video_path:
        print("Đã hủy chọn file.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    # 2. Tạo giao diện Trackbar
    cv2.namedWindow('HSV_Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV_Trackbars', 400, 300)
    
    cv2.createTrackbar('H_MIN', 'HSV_Trackbars', 90, 179, nothing)
    cv2.createTrackbar('S_MIN', 'HSV_Trackbars', 50, 255, nothing)
    cv2.createTrackbar('V_MIN', 'HSV_Trackbars', 40, 255, nothing)
    cv2.createTrackbar('H_MAX', 'HSV_Trackbars', 135, 179, nothing)
    cv2.createTrackbar('S_MAX', 'HSV_Trackbars', 255, 255, nothing)
    cv2.createTrackbar('V_MAX', 'HSV_Trackbars', 255, 255, nothing)

    print("--- HƯỚNG DẪN ---")
    print("1. Kéo các thanh trượt để phần mặt đường hiển thị màu TRẮNG, nhiễu hiển thị màu ĐEN.")
    print("2. Nhấn phím SPACE để xem khung hình (Frame) tiếp theo của video.")
    print("3. Nhấn phím 'q' để thoát và in ra thông số cuối cùng.")

    ret, frame = cap.read()
    
    while True:
        if not ret or frame is None:
            # Hết video thì lặp lại từ đầu
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            continue

        # Cắt lấy ảnh gốc (nếu video là HStack 1280x480, ảnh gốc nằm nửa bên trái)
        if frame.shape[1] > 640:
            roi = frame[:, :640]
        else:
            roi = frame.copy()
            
        # Tiền xử lý
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Đọc trackbar
        h_min = cv2.getTrackbarPos('H_MIN', 'HSV_Trackbars')
        s_min = cv2.getTrackbarPos('S_MIN', 'HSV_Trackbars')
        v_min = cv2.getTrackbarPos('V_MIN', 'HSV_Trackbars')
        h_max = cv2.getTrackbarPos('H_MAX', 'HSV_Trackbars')
        s_max = cv2.getTrackbarPos('S_MAX', 'HSV_Trackbars')
        v_max = cv2.getTrackbarPos('V_MAX', 'HSV_Trackbars')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Lọc màu
        mask = cv2.inRange(hsv, lower, upper)
        
        # Lớp phủ xanh
        green_layer = np.zeros_like(roi)
        green_layer[mask == 255] = [0, 255, 0]
        preview = cv2.addWeighted(roi, 1.0, green_layer, 0.5, 0)

        cv2.imshow('Original + Mask Preview', preview)
        cv2.imshow('Black/White Mask', mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\n=== THÔNG SỐ HSV CHO MẶT ĐƯỜNG (V3) ===")
            print(f"self.H_MIN, self.S_MIN, self.V_MIN = {h_min}, {s_min}, {v_min}")
            print(f"self.H_MAX, self.S_MAX, self.V_MAX = {h_max}, {s_max}, {v_max}")
            print("=========================================\n")
            break
        elif key == ord(' '): # Nhấn phím cách để qua frame
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_calibrator()
