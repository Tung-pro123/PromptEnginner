import os
from ultralytics import YOLO

def train_and_export():
    print("=== Khởi tạo mô hình YOLOv8 Nano Segmentation ===")
    model = YOLO("yolov8n-seg.pt") # Sử dụng model Nano để đảm bảo FPS cao trên Jetson

    # Đường dẫn tuyệt đối tới file data.yaml
    dataset_path = r"d:\FPT_University\JetsonAIRacer\PromptEnginner\src\smart_city\dataset\Smart city 2.v2i.yolov8\data.yaml"
    
    print(f"=== Bắt đầu huấn luyện với dataset: {dataset_path} ===")
    
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CẢNH BÁO: Máy bạn chưa cài đặt PyTorch phiên bản hỗ trợ GPU (CUDA). Quá trình train sẽ chạy bằng CPU và RẤT CHẬM!")
        print("Vui lòng cài đặt bản PyTorch CUDA (ví dụ: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)")
    
    # Bắt đầu quá trình huấn luyện
    # Đã cấu hình imgsz=320 để giảm dung lượng, device=0 để ép chạy GPU
    results = model.train(
        data=dataset_path,
        epochs=100,         # Số lượng vòng lặp qua dataset
        imgsz=320,          # Kích thước ảnh đầu vào. Rất quan trọng để giữ FPS cao trên xe
        batch=16,           # Số lượng ảnh xử lý cùng lúc (tuỳ RAM GPU)
        device=device,      # Tự động chọn GPU nếu có, ngược lại CPU
        patience=20,        # Dừng sớm nếu sau 20 epoch không cải thiện
        project="runs/segment",
        name="smart_city_lane_crosswalk"
    )

    print("=== Quá trình Huấn luyện Hoàn tất! ===")
    
    # Lấy đường dẫn tới file mô hình tốt nhất (best.pt)
    best_weights_path = os.path.join(results.save_dir, "weights", "best.pt")
    
    print(f"=== Bắt đầu xuất (Export) mô hình {best_weights_path} sang dạng ONNX ===")
    
    # Tải lại mô hình tốt nhất vừa train xong
    best_model = YOLO(best_weights_path)
    
    # Export sang dạng ONNX. Lưu ý: có thể dùng ONNX trên Jetson, 
    # nhưng lý tưởng nhất là chạy thêm lệnh export sang TensorRT (.engine) trực tiếp trên Jetson.
    onnx_path = best_model.export(format="onnx", imgsz=320, opset=12)
    
    print(f"=== Export hoàn tất! File ONNX được lưu tại: {onnx_path} ===")
    print("Bạn có thể mang file .onnx hoặc .pt này sang xe Jetson để chạy.")

if __name__ == "__main__":
    train_and_export()
