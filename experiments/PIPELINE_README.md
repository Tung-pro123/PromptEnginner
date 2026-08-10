# Hướng Dẫn Sử Dụng Pipeline Triển Khai (Train -> Jetson)

Tài liệu này hướng dẫn bạn quy trình chuẩn từ khi chuẩn bị dữ liệu, huấn luyện mô hình cho đến khi mang lên NVIDIA Jetson chạy thực tế.

## 1. Chuẩn Bị Dữ Liệu
Chạy script giải nén và cắt ảnh từ video (`.avi`) sang tập hợp các ảnh (`.jpg`):
```powershell
python e:\robot-jeston\experiments\extract_frames.py
```
> **Lưu ý:** Tool này sẽ cắt vùng ROI (nửa dưới của ảnh camera) và lưu lại dưới định dạng Grayscale (hoặc Color tùy thiết lập).

## 2. Huấn Luyện Giai Đoạn 1: Autoencoder (Tái tạo ảnh)
Ép mạng CNN học cách trích xuất đặc trưng của vạch kẻ đường thông qua việc nén và giải nén ảnh.
```powershell
python e:\robot-jeston\experiments\train_stage1.py
```
> Trọng số sẽ được lưu tại `weights/autoencoder_only.pth`. Càng train lâu, đường nét vạch kẻ đứt được tái tạo càng rõ.

## 3. Huấn Luyện Giai Đoạn 2: Control Predictor (Bẻ lái)
Đóng băng bộ nén (Encoder), chỉ học cách map từ Latent Vector `z` sang góc lái (Steer) và ga (Throttle).
```powershell
python e:\robot-jeston\experiments\train_stage2.py
```
> Trọng số hoàn chỉnh lưu tại `weights/vision_autoencoder.pth`.

## 4. Mô Phỏng & Đánh Giá (Test)
Chạy thử nghiệm trên tập dữ liệu để xem biểu đồ Steering/Throttle và xem ảnh tái tạo của AI.
```powershell
python e:\robot-jeston\experiments\test_vision_continuous.py
```
> Kết quả được lưu tại thư mục `plots/`.

---

## 5. Xuất Mô Hình Để Đưa Lên Xe (Export ONNX)
Khi chạy thực tế, chúng ta **không cần** bộ Decoder (khối giải nén ảnh mất rất nhiều thời gian). 
Chạy script sau để loại bỏ Decoder và xuất phần còn lại ra định dạng chuẩn `ONNX`:
```powershell
python e:\robot-jeston\experiments\export_to_onnx.py
```
> Output: `weights/vision_inference.onnx` (File này cực nhẹ).

---

## 6. Biên Dịch Sang TensorRT Trên Jetson
**⚠️ BƯỚC NÀY BẮT BUỘC PHẢI CHẠY TRỰC TIẾP TRÊN XE JETSON ⚠️**

1. Copy file `vision_inference.onnx` vào thiết bị Jetson.
2. Mở Terminal trên Jetson.
3. Chạy lệnh sau để chuyển đổi ONNX sang file TensorRT Engine (`.engine` hoặc `.trt`). Việc bật cờ `--fp16` sẽ giúp model chạy với độ chính xác bán thập phân, tăng gấp đôi tốc độ FPS (Frames Per Second) mà hầu như không giảm độ chuẩn xác.

```bash
# Câu lệnh chuẩn sử dụng trtexec (có sẵn trong bộ cài JetPack của Jetson)
/usr/src/tensorrt/bin/trtexec \
    --onnx=vision_inference.onnx \
    --saveEngine=vision_inference_fp16.engine \
    --fp16 \
    --workspace=1024
```

> **Sau khi có file `.engine`:** Trong source code ROS hoặc Python chạy trên Jetson, bạn dùng thư viện `tensorrt` để load file engine này lên và suy luận (inference) bình thường! Tốc độ hứa hẹn sẽ đạt >60 FPS!
