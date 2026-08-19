import os
import sys
import cv2
import glob
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.smart_city.config import SmartCityConfig
from src.smart_city.perception.traffic_detector import TrafficDetector

def main():
    print("=== Khởi tạo TrafficDetector ===")
    cfg = SmartCityConfig()
    sign_detector = TrafficDetector(cfg.image_width, cfg.image_height)
    
    # Lấy danh sách ảnh trong tập test sign
    sign_img_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'sign')
    images = glob.glob(os.path.join(sign_img_dir, "*.jpg"))
    
    if not images:
        print("Không tìm thấy ảnh test trong thư mục dataset/sign!")
        return
        
    output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Bắt đầu test 5 ảnh ngẫu nhiên trong thư mục sign...")
    
    # Chọn random 5 ảnh
    random.shuffle(images)
    test_images = images[:5]
    
    for i, img_path in enumerate(test_images): 
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Resize ảnh về kích thước thật của Jetbot
        img = cv2.resize(img, (cfg.image_width, cfg.image_height))
        
        # Chạy thuật toán TrafficDetector
        light, sign = sign_detector.detect(img)
        
        # Vẽ giao diện Debug
        cv2.putText(img, f"DETECTED: {sign}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if sign != "NONE" else (0, 255, 0), 2)
        
        # Vẽ khung ROI (vùng quét biển báo)
        cv2.rectangle(img, (0, 0), (cfg.image_width, sign_detector.roi_y_end), (255, 0, 0), 2)
        cv2.putText(img, "ROI", (5, sign_detector.roi_y_end - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Lưu kết quả
        out_path = os.path.join(output_dir, f"sign_test_{i+1}.jpg")
        cv2.imwrite(out_path, img)
        print(f"[{i+1}/5] File: {img_name[:15]}... -> Sign: {sign}")

if __name__ == '__main__':
    main()
