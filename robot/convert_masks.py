import os
import cv2
import glob
import shutil
import yaml
import sys
import numpy as np

def mask_to_yolo(mask_path, txt_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return False
    
    H, W = mask.shape
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open(txt_path, 'w') as f:
        for contour in contours:
            if cv2.contourArea(contour) < 100: continue
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            flattened = approx.flatten()
            if len(flattened) < 6: continue
            
            normalized = []
            for i in range(0, len(flattened), 2):
                x = flattened[i] / W
                y = flattened[i+1] / H
                normalized.append(f"{x:.5f} {y:.5f}")
                
            f.write(f"0 {' '.join(normalized)}\n")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Vui lòng truyền tên thư mục dataset vừa giải nén.")
        print("Ví dụ: python convert_masks.py my-first-project-1")
        sys.exit(1)
        
    dataset_dir = sys.argv[1]
    if not os.path.exists(dataset_dir):
        print(f"Không tìm thấy thư mục: {dataset_dir}")
        sys.exit(1)

    print(f"Bắt đầu convert PNG Masks sang định dạng YOLOv8 Segmentation tại: {dataset_dir}")

    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir): continue
        
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        images_dir = os.path.join(dataset_dir, split, 'images')
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        mask_files = glob.glob(os.path.join(split_dir, '*_mask.png'))
        count = 0
        for mask_file in mask_files:
            base_name = os.path.basename(mask_file).replace('_mask.png', '')
            
            img_file_jpg = os.path.join(split_dir, base_name + '.jpg')
            img_file_png = os.path.join(split_dir, base_name + '.png')
            img_file = img_file_jpg if os.path.exists(img_file_jpg) else img_file_png
            
            if not os.path.exists(img_file): continue
                
            txt_file = os.path.join(labels_dir, base_name + '.txt')
            
            if mask_to_yolo(mask_file, txt_file):
                shutil.move(img_file, os.path.join(images_dir, os.path.basename(img_file)))
                count += 1
            os.remove(mask_file)
        print(f"Đã convert xong {count} ảnh trong phần {split}")

    # Tạo data.yaml
    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {0: 'drivable_area'}
    }

    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nHOÀN TẤT! Dataset đã được cấu hình tại: {yaml_path}")
    print(f"Lệnh để train: python train_yolo.py {yaml_path}")
