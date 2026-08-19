import os
import cv2
import numpy as np
import glob
import shutil

# --- BƯỚC 1: TẢI DATASET TỪ ROBOFLOW ---
print("Đang tải dataset từ Roboflow...")
try:
    from roboflow import Roboflow
    import yaml
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "pyyaml"])
    from roboflow import Roboflow
    import yaml

rf = Roboflow(api_key="UR9RR91iCc6NifX1tI2B")
project = rf.workspace("hackathon2025-acu9o").project("my-first-project-fwmey")
version = project.version(1)
dataset = version.download("png-mask-semantic")

dataset_dir = dataset.location
print(f"Đã tải dataset về: {dataset_dir}")

# --- BƯỚC 2: CONVERT MASK PNG SANG YOLOv8 POLYGON ---
print("Bắt đầu convert PNG Masks sang định dạng YOLOv8 Segmentation...")

def mask_to_yolo(mask_path, txt_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return False
    
    H, W = mask.shape
    # Dùng ngưỡng để nhị phân hoá (mask thường là 255 cho vùng label, 0 cho nền)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Tìm viền của vùng màu trắng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open(txt_path, 'w') as f:
        for contour in contours:
            # Bỏ qua các contour quá nhỏ (nhiễu)
            if cv2.contourArea(contour) < 100: continue
            
            # Làm mịn contour để giảm bớt số lượng điểm (giúp model chạy nhanh hơn)
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ép mảng về dạng 1 chiều [x1, y1, x2, y2...] và chuẩn hoá về [0-1]
            flattened = approx.flatten()
            if len(flattened) < 6: continue # Cần ít nhất 3 điểm để tạo thành đa giác
            
            normalized = []
            for i in range(0, len(flattened), 2):
                x = flattened[i] / W
                y = flattened[i+1] / H
                normalized.append(f"{x:.5f} {y:.5f}")
                
            # Ghi ra file với Class 0 (Road)
            f.write(f"0 {' '.join(normalized)}\n")
    return True

# Lặp qua các thư mục train, valid, test
for split in ['train', 'valid', 'test']:
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir): continue
    
    # Tạo thư mục labels
    labels_dir = os.path.join(dataset_dir, split, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    # Move ảnh vào thư mục images cho chuẩn cấu trúc YOLO
    images_dir = os.path.join(dataset_dir, split, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Quét tất cả ảnh mask
    mask_files = glob.glob(os.path.join(split_dir, '*-mask.png'))
    for mask_file in mask_files:
        base_name = os.path.basename(mask_file).replace('-mask.png', '')
        
        # Tìm ảnh gốc tương ứng (có thể là .jpg hoặc .png)
        img_file_jpg = os.path.join(split_dir, base_name + '.jpg')
        img_file_png = os.path.join(split_dir, base_name + '.png')
        img_file = img_file_jpg if os.path.exists(img_file_jpg) else img_file_png
        
        if not os.path.exists(img_file): continue
            
        txt_file = os.path.join(labels_dir, base_name + '.txt')
        
        # Convert mask sang txt
        if mask_to_yolo(mask_file, txt_file):
            # Di chuyển ảnh gốc vào thư mục images
            shutil.move(img_file, os.path.join(images_dir, os.path.basename(img_file)))
            
        # Xóa file mask đi cho nhẹ
        os.remove(mask_file)

# --- BƯỚC 3: TẠO FILE data.yaml CHUẨN YOLOv8 ---
yaml_content = {
    'path': dataset_dir,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'names': {0: 'drivable_area'}
}

yaml_path = os.path.join(dataset_dir, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

print(f"HOÀN TẤT! Dataset đã được convert và cấu hình tại: {yaml_path}")
print("Bây giờ bạn có thể mở Terminal và chạy lệnh:")
print(f"python train_yolo.py {yaml_path}")
