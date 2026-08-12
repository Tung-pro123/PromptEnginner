import os
import glob
import cv2
import pandas as pd
from tqdm import tqdm

TARGETS = ['steer_raw', 'target_speed']
IMG_WIDTH = 128
IMG_HEIGHT = 32

def extract_frames():
    logs_dir = r"e:\robot-jeston\logs\logs"
    dataset_dir = r"e:\robot-jeston\logs\dataset"
    images_dir = os.path.join(dataset_dir, "images")
    
    os.makedirs(images_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(logs_dir, "*.csv"))
    
    all_data = []
    
    for csv_path in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        avi_path = os.path.join(logs_dir, f"{base_name}.avi")
        
        if not os.path.exists(avi_path):
            continue
            
        print(f"Đang xử lý: {base_name}")
        
        df = pd.read_csv(csv_path)
        # Bỏ qua các hàng bị thiếu nhãn quan trọng (sẽ không nội suy để đảm bảo dữ liệu thật)
        df = df.dropna(subset=TARGETS)
        
        # (Đã tắt) Vuốt mượt nhãn theo yêu cầu: Giữ nguyên dữ liệu thô
        # for target_col in TARGETS:
        #     df[target_col] = df[target_col].rolling(window=7, min_periods=1, center=True).mean()
            
        cap = cv2.VideoCapture(avi_path)
        
        frame_idx = 0
        valid_indices = set(df.index.tolist())
        
        # Đọc video tuần tự (rất nhanh so với việc nhảy frame)
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in valid_indices:
                # 1. Cắt lấy ROI theo đúng viền vàng từ nửa trái video (Y từ 144 đến 480, X từ 0 đến 640)
                # Vì video gốc là 1280x480 (chứa cả 2 nửa), ta chỉ lấy nửa trái (0:640)
                roi = frame[144:480, 0:640]
                
                # 2. Resize
                img = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
                
                # Chuyển sang ảnh xám (Grayscale)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Lưu ảnh ra đĩa
                img_name = f"{base_name}_frame_{frame_idx:06d}.jpg"
                img_path = os.path.join(images_dir, img_name)
                cv2.imwrite(img_path, img)
                
                # Lấy dữ liệu nhãn tương ứng
                row = df.loc[frame_idx]
                all_data.append({
                    'image_path': img_name,
                    'trip_id': base_name, # Giữ ID chuyến đi để tạo sequence không bị nối chéo
                    'steer_raw': row['steer_raw'],
                    'target_speed': row['target_speed']
                })
                
            frame_idx += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
    # Lưu lại file CSV tổng hợp cho toàn bộ dataset
    final_df = pd.DataFrame(all_data)
    final_csv_path = os.path.join(dataset_dir, "dataset_labels.csv")
    final_df.to_csv(final_csv_path, index=False)
    
    print(f"\n✅ Hoàn thành! Đã giải nén {len(final_df)} tấm ảnh ROI ra {images_dir}")
    print(f"File nhãn tổng hợp lưu tại: {final_csv_path}")

if __name__ == "__main__":
    extract_frames()
