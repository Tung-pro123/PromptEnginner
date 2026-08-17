import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import sys
import math

def nothing(x):
    pass

def run_calibrator():
    # 1. Chọn file video log
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw() # Ẩn cửa sổ chính
        print("Hãy chọn file video .avi (VD: speed_racing_v2_...avi) để lấy mẫu màu...")
        video_path = filedialog.askopenfilename(
            title="Chọn file Video Log",
            filetypes=[("Video Files", "*.mp4 *.avi"), ("All Files", "*.*")]
        )
        
    if not video_path:
        print("Đã hủy chọn file.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    # 2. Tạo giao diện Trackbar (Mở rộng cho V3.1)
    cv2.namedWindow('V3.1_Calibrator', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('V3.1_Calibrator', 450, 600)
    
    # HSV Trackbars
    cv2.createTrackbar('H_MIN', 'V3.1_Calibrator', 0, 179, nothing)
    cv2.createTrackbar('S_MIN', 'V3.1_Calibrator', 0, 255, nothing)
    cv2.createTrackbar('V_MIN', 'V3.1_Calibrator', 0, 255, nothing)
    cv2.createTrackbar('H_MAX', 'V3.1_Calibrator', 179, 179, nothing)
    cv2.createTrackbar('S_MAX', 'V3.1_Calibrator', 255, 255, nothing)
    cv2.createTrackbar('V_MAX', 'V3.1_Calibrator', 255, 255, nothing)
    
    # V3.1 Advanced Trackbars
    cv2.createTrackbar('Blur_Size', 'V3.1_Calibrator', 5, 21, nothing) # Chỉ số lẻ
    cv2.createTrackbar('ROI_Y_Start (%)', 'V3.1_Calibrator', 45, 100, nothing)
    cv2.createTrackbar('LAB_A_MIN', 'V3.1_Calibrator', 0, 255, nothing)
    cv2.createTrackbar('Horizon_Y1', 'V3.1_Calibrator', 152, 480, nothing)
    cv2.createTrackbar('Horizon_Y2', 'V3.1_Calibrator', 234, 480, nothing)
    
    # NEW: Pixel & Angle Thresholds (Dùng FitLine)
    cv2.createTrackbar('Pix_Thresh', 'V3.1_Calibrator', 50, 500, nothing)
    cv2.createTrackbar('Angle_Thresh', 'V3.1_Calibrator', 15, 90, nothing) # Góc lệch tối đa (Độ)
    cv2.createTrackbar('Center_Zone', 'V3.1_Calibrator', 120, 320, nothing) # Khóa line giữa

    print("--- HƯỚNG DẪN V3.1 CALIBRATOR (CHẾ ĐỘ FITLINE KẺ ĐƯỜNG) ---")
    print("1. Kéo HSV để lọc màu vạch kẻ đường.")
    print("2. Hệ thống sẽ 'kẻ một đường thẳng' nối các điểm ảnh trong vùng Horizon.")
    print("3. Nếu đường kẻ này SONG SONG với trục dọc (Góc lệch < Angle_Thresh) -> STRAIGHT.")
    print("4. Nhấn SPACE để qua Frame tiếp theo, 'q' để Thoát và In thông số.")

    ret, frame = cap.read()
    
    while True:
        if not ret or frame is None:
            # Hết video thì lặp lại từ đầu
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            continue

        # Cắt lấy ảnh gốc (nếu video là HStack 1280x480)
        if frame.shape[1] > 640:
            roi = frame[:, :640].copy()
        else:
            roi = frame.copy()
            
        height, width = roi.shape[:2]

        # Đọc tham số từ Trackbars
        h_min = cv2.getTrackbarPos('H_MIN', 'V3.1_Calibrator')
        s_min = cv2.getTrackbarPos('S_MIN', 'V3.1_Calibrator')
        v_min = cv2.getTrackbarPos('V_MIN', 'V3.1_Calibrator')
        h_max = cv2.getTrackbarPos('H_MAX', 'V3.1_Calibrator')
        s_max = cv2.getTrackbarPos('S_MAX', 'V3.1_Calibrator')
        v_max = cv2.getTrackbarPos('V_MAX', 'V3.1_Calibrator')
        
        blur_sz = cv2.getTrackbarPos('Blur_Size', 'V3.1_Calibrator')
        if blur_sz % 2 == 0: blur_sz += 1 # Đảm bảo số lẻ
        if blur_sz < 1: blur_sz = 1
        
        roi_y_pct = cv2.getTrackbarPos('ROI_Y_Start (%)', 'V3.1_Calibrator') / 100.0
        lab_a_min = cv2.getTrackbarPos('LAB_A_MIN', 'V3.1_Calibrator')
        
        hy1 = cv2.getTrackbarPos('Horizon_Y1', 'V3.1_Calibrator')
        hy2 = cv2.getTrackbarPos('Horizon_Y2', 'V3.1_Calibrator')
        pix_thresh = cv2.getTrackbarPos('Pix_Thresh', 'V3.1_Calibrator')
        angle_thresh = cv2.getTrackbarPos('Angle_Thresh', 'V3.1_Calibrator')
        center_zone = cv2.getTrackbarPos('Center_Zone', 'V3.1_Calibrator')
        
        if hy1 > hy2: hy1, hy2 = hy2, hy1

        # ==========================================================
        # MÔ PHỎNG CHẾ ĐỘ FITLINE (Kẻ đường nối điểm ảnh)
        # ==========================================================
        # Định nghĩa ngưỡng HSV từ Trackbars
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        
        horizon_roi = roi[hy1:hy2, :].copy()
        
        # Tiền xử lý vùng horizon
        h_blur = cv2.GaussianBlur(horizon_roi, (blur_sz, blur_sz), 0)
        h_hsv = cv2.cvtColor(h_blur, cv2.COLOR_BGR2HSV)
        h_mask = cv2.inRange(h_hsv, lower_hsv, upper_hsv)
        
        if lab_a_min > 0:
            h_lab = cv2.cvtColor(h_blur, cv2.COLOR_BGR2LAB)
            h_a = h_lab[:, :, 1]
            h_lab_mask = cv2.inRange(h_a, lab_a_min, 255)
            h_mask = cv2.bitwise_and(h_mask, h_lab_mask)
            
        pixel_count = cv2.countNonZero(h_mask)
        
        horizon_state = "UNKNOWN"
        box_color = (0, 255, 255)
        angle_deg = 0.0
        line_pts = None
        
        if pixel_count > pix_thresh:
            # TÌM CÁC ĐƯỜNG RIÊNG BIỆT (Contours)
            contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Lọc các contour quá nhỏ
                valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
                
                if valid_contours:
                    # Tìm contour gần TRỤC GIỮA nhất (để lấy "line giữa")
                    img_center = width / 2.0
                    best_contour = None
                    min_dist = float('inf')
                    
                    for c in valid_contours:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            dist = abs(cx - img_center)
                            
                            # NGĂN CHẶN NHẢY SANG LINE BIÊN: 
                            # Chỉ lấy những line nằm trong khu vực Center_Zone
                            if dist > center_zone:
                                continue
                                
                            if dist < min_dist:
                                min_dist = dist
                                best_contour = c
                    
                    if best_contour is not None:
                        # Kẻ đường thẳng qua "Line Giữa" này
                        [vx, vy, x, y] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        vx = float(vx[0])
                        vy = float(vy[0])
                        x = float(x[0])
                        y = float(y[0])
                        
                        # Tính góc lệch so với trục dọc (Trục Y)
                        # vy = 1, vx = 0 -> song song trục dọc -> góc = 0
                        if vy == 0:
                            angle_deg = 90.0
                        else:
                            angle_deg = math.degrees(math.atan(abs(vx / vy)))
                        
                        # Tính toán tọa độ để vẽ đường thẳng
                        h_height = hy2 - hy1
                        top_x = int(x + (0 - y) * (vx / vy)) if vy != 0 else int(x)
                        bot_x = int(x + (h_height - y) * (vx / vy)) if vy != 0 else int(x)
                        
                        # Fix: Clip to prevent OpenCV overflow on nearly horizontal lines
                        top_x = max(-10000, min(10000, top_x))
                        bot_x = max(-10000, min(10000, bot_x))
                        
                        line_pts = ((int(top_x), int(hy1)), (int(bot_x), int(hy2)))
                        
                        # So sánh góc với Angle_Thresh
                        if angle_deg > angle_thresh:
                            horizon_state = "CURVE"
                            box_color = (0, 0, 255) # Đỏ
                        else:
                            horizon_state = "STRAIGHT"
                            box_color = (0, 255, 0) # Xanh lá
                else:
                    horizon_state = "CURVE (No Valid Line)"
                    box_color = (0, 0, 255)
        else:
            horizon_state = "CURVE (No Pixels)"
            box_color = (0, 0, 255) # Đỏ
            
        # ==========================================================
        # VẼ HIỂN THỊ (PREVIEW)
        # ==========================================================
        preview = roi.copy()
        
        # 1. Vẽ ROI chính
        roi_y_start = int(height * roi_y_pct)
        cv2.line(preview, (0, roi_y_start), (width, roi_y_start), (0, 0, 255), 2)
        cv2.putText(preview, "Main ROI Start", (10, roi_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 2. Phủ màu tím lên mask
        tint = np.zeros_like(horizon_roi)
        tint[h_mask == 255] = [255, 0, 255]
        preview[hy1:hy2, :] = cv2.addWeighted(preview[hy1:hy2, :], 0.6, tint, 0.4, 0)
        
        # 3. Vẽ Horizon Box
        cv2.rectangle(preview, (0, hy1), (width, hy2), box_color, 2)
        
        # 4. Vẽ Trục Giữa màu vàng và Vùng Khóa Center Zone (Màu xám)
        cv2.line(preview, (int(width/2), hy1), (int(width/2), hy2), (0, 255, 255), 1)
        
        left_bound = int(width/2 - center_zone)
        right_bound = int(width/2 + center_zone)
        cv2.line(preview, (left_bound, hy1), (left_bound, hy2), (128, 128, 128), 1)
        cv2.line(preview, (right_bound, hy1), (right_bound, hy2), (128, 128, 128), 1)
        
        # 5. KẺ ĐƯỜNG MÀU XANH/ĐỎ NỐI CÁC ĐIỂM ẢNH LẠI
        if line_pts is not None:
            cv2.line(preview, line_pts[0], line_pts[1], box_color, 3)
        
        # 6. Ghi Text Status
        cv2.putText(preview, f"Horizon: {horizon_state}", (10, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.putText(preview, f"Pix: {pixel_count}  Angle: {angle_deg:.1f} (Thresh: {angle_thresh})", 
                    (width - 320, hy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        cv2.imshow('V3.1 SYNCHRONIZED CALIBRATOR', preview)
        cv2.imshow('Horizon Scanner MASK', h_mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\n=== THÔNG SỐ CẤU HÌNH CHO V3.1 CONFIG.PY ===")
            print(f"cfg.H_MIN, cfg.S_MIN, cfg.V_MIN = {h_min}, {s_min}, {v_min}")
            print(f"cfg.H_MAX, cfg.S_MAX, cfg.V_MAX = {h_max}, {s_max}, {v_max}")
            print(f"cfg.blur_size = {blur_sz}")
            print(f"cfg.roi_y_start = {roi_y_pct:.2f}")
            print(f"cfg.use_lab_constraint = True")
            print(f"cfg.lab_a_min = {lab_a_min}")
            print(f"cfg.horizon_scan_y_start = {hy1}")
            print(f"cfg.horizon_scan_y_end = {hy2}")
            print(f"cfg.horizon_angle_thresh = {angle_thresh}")
            print(f"cfg.horizon_center_zone = {center_zone}")
            print("=============================================\n")
            break
        elif key == ord(' '): # Nhấn phím cách để qua frame
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_calibrator()
