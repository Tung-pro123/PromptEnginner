"""
Script huấn luyện mô hình YOLOv8 Segmentation cho xe JetRacer.
Yêu cầu cài đặt: pip install ultralytics
"""
from ultralytics import YOLO
import sys

def train_model(data_yaml_path):
    print("Khởi tạo mô hình YOLOv8n-seg (phiên bản nhỏ gọn và nhanh nhất)...")
    # Load mô hình pre-trained
    model = YOLO('yolov8n-seg.pt')
    
    print(f"Bắt đầu huấn luyện với cấu hình: {data_yaml_path}")
    # Huấn luyện mô hình
    results = model.train(
        data=data_yaml_path,
        epochs=50,           # Train 50 vòng (có thể tăng lên 100 nếu GPU mạnh)
        imgsz=320,           # Resize ảnh về 320x320 để train nhanh và chạy nhẹ trên Jetson
        batch=16,            # Batch size
        device='cpu',        # Dùng CPU vì máy không có GPU (hoặc GPU chưa cài CUDA)
        name='jetson_track_seg'
    )
    
    print("Huấn luyện hoàn tất! Mô hình tốt nhất được lưu tại: runs/segment/jetson_track_seg/weights/best.pt")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Cách dùng: python train_yolo.py <đường_dẫn_tới_file_data.yaml>")
        print("Ví dụ: python train_yolo.py dataset/data.yaml")
        sys.exit(1)
        
    train_model(sys.argv[1])
