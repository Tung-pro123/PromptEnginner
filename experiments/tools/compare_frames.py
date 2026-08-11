import os
import glob
import csv
import sys

try:
    import cv2
except ImportError:
    print("Thư viện 'cv2' (opencv-python) chưa được cài đặt. Vui lòng chạy: pip install opencv-python")
    sys.exit(1)

def count_csv_rows(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Sử dụng csv reader để đếm dòng
        reader = csv.reader(f)
        # Bỏ qua header
        next(reader, None)
        return sum(1 for row in reader)

def count_avi_frames(avi_path):
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return -1
    # Đọc tổng số frame
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def compare_frames():
    logs_dir = r"e:\robot-jeston\logs\logs"
    
    csv_files = glob.glob(os.path.join(logs_dir, "*.csv"))
    
    print(f"{'Tên file (Base)':<30} | {'Số dòng CSV':<15} | {'Số frame AVI':<15} | {'Chênh lệch (CSV - AVI)':<25}")
    print("-" * 95)
    
    for csv_path in sorted(csv_files):
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        avi_path = os.path.join(logs_dir, f"{base_name}.avi")
        
        # Số dòng dữ liệu trong file CSV (trừ header)
        csv_rows = count_csv_rows(csv_path)
        
        # Số frame trong file AVI tương ứng
        if os.path.exists(avi_path):
            avi_frames = count_avi_frames(avi_path)
            if avi_frames >= 0:
                diff = csv_rows - avi_frames
                diff_str = f"{diff:+} frames"
            else:
                avi_frames = "Lỗi đọc file"
                diff_str = "N/A"
        else:
            avi_frames = "Không có file"
            diff_str = "N/A"
            
        print(f"{base_name:<30} | {csv_rows:<15} | {avi_frames if isinstance(avi_frames, str) else avi_frames:<15} | {diff_str:<25}")

if __name__ == "__main__":
    compare_frames()
