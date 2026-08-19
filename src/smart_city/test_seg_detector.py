import os
import sys
import cv2
import glob

# Đảm bảo đường dẫn tới thư mục src gốc để import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.smart_city.config import SmartCityConfig
from src.smart_city.perception.seg_lane_detector import YoloSegDetector

def main():
    print("=== Khởi tạo YoloSegDetector ===")
    cfg = SmartCityConfig()
    # Chuyển sang dùng file .pt để test trên máy tính tránh lỗi ONNXRuntime thiếu DLL CUDA 13
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best.pt')
    
    detector = YoloSegDetector(model_path, cfg)
    
    # Lấy danh sách ảnh trong tập test
    test_img_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'Smart city 2.v2i.yolov8', 'test', 'images')
    images = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    
    if not images:
        print("Không tìm thấy ảnh test!")
        return
        
    # Tạo thư mục lưu kết quả
    output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Bắt đầu test 5 ảnh ngẫu nhiên trong tập test...")
    
    for img_path in images[:5]: 
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Resize ảnh về kích thước thật của Jetbot
        img = cv2.resize(img, (cfg.image_width, cfg.image_height))
        
        # Chạy thuật toán
        res = detector.detect(img)
        
        # Vẽ giao diện Debug với các Mask
        debug_img = detector.draw_debug(img, res)
        
        # Lưu kết quả
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, debug_img)
        print(f"Đã lưu kết quả tại: {out_path}")
        print(f"  -> Mode: {res.mode}, Lệch tâm (Setpoint): {res.center_x}, Có Crosswalk: {res.has_crosswalk}")

if __name__ == '__main__':
    main()
