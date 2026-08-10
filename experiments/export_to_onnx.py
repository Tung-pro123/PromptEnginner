import os
import torch
import torch.nn as nn
from train_vision_autoencoder import VisionAutoencoder, LATENT_DIM, IMG_HEIGHT, IMG_WIDTH

# Định nghĩa một class gộp chung Encoder và Predictor lại với nhau
# Mô hình này loại bỏ hoàn toàn Decoder, tối ưu tốc độ và dung lượng khi chạy trên Jetson
class VisionInferenceModel(nn.Module):
    def __init__(self, encoder, predictor):
        super(VisionInferenceModel, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        
    def forward(self, x):
        # x shape: (Batch, Channels, H, W)
        z = self.encoder(x)
        out = self.predictor(z)
        # return tensor 1D: [steer, throttle]
        return out

def export_onnx():
    device = torch.device("cpu") # Xuất ONNX nên dùng CPU cho dễ tương thích
    
    # 1. Khởi tạo mô hình đầy đủ và load trọng số
    model_full = VisionAutoencoder(LATENT_DIM)
    
    weights_path = r"e:\robot-jeston\experiments\weights\vision_autoencoder.pth"
    if not os.path.exists(weights_path):
        print(f"Lỗi: Không tìm thấy trọng số tại {weights_path}")
        return
        
    model_full.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model_full.eval()
    print("Successfully loaded PyTorch weights.")
    
    # 2. Tạo Inference Model bằng cách rút ruột Encoder và Predictor
    inference_model = VisionInferenceModel(model_full.encoder, model_full.predictor)
    inference_model.eval()
    
    # 3. Tạo một tensor dummy giả lập 1 bức ảnh grayscale đi vào
    # Batch = 1, Channels = 1 (Grayscale), H = 32, W = 128
    dummy_input = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH, device=device)
    
    # 4. Xuất ra ONNX
    onnx_path = r"e:\robot-jeston\experiments\weights\vision_inference.onnx"
    
    print("Compiling to ONNX format...")
    torch.onnx.export(
        inference_model,               # Mô hình Pytorch
        dummy_input,                   # Đầu vào giả lập
        onnx_path,                     # Đường dẫn file xuất ra
        export_params=True,            # Kèm trọng số
        opset_version=12,              # Chuẩn Opset thông dụng
        do_constant_folding=True,      # Tối ưu hóa các tính toán hằng số
        input_names=['input_image'],   # Đặt tên tensor đầu vào
        output_names=['control_out'],  # Đặt tên tensor đầu ra
        dynamic_axes={
            'input_image': {0: 'batch_size'},    # Khai báo kích thước batch có thể thay đổi
            'control_out': {0: 'batch_size'}
        }
    )
    
    print(f"Success! Saved lightweight Inference ONNX model at: {onnx_path}")
    print("You can now move this .onnx file to Jetson and use trtexec to convert it to TensorRT.")

if __name__ == "__main__":
    export_onnx()
