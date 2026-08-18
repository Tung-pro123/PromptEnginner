#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from robot.ai.sign_color_detector import SignColorDetector

class CNNSignDetector:
    """
    Lớp vỏ nhận diện biển báo sử dụng mạng CNN.
    Phát triển dựa trên SignRecognizerCNN trong thư mục robot/ai nhưng được 
    thiết kế để trả về dữ liệu có cấu trúc cho FSM điều khiển thay vì chỉ vẽ lên ảnh.
    Sẽ load model dạng ONNX/TensorRT theo yêu cầu tối ưu của người dùng.
    """
    def __init__(self, model_path="models/sign_recognizer.onnx", threshold=0.90):
        self.model_path = model_path
        self.threshold = threshold
        self.color_detector = SignColorDetector()
        
        # Các class do model CNN dự đoán (bao gồm cả đèn giao thông và biển cấm)
        self.classes = [
            'forbidden', 
            'left', 
            'right', # Thêm right nếu model của bạn có hỗ trợ, tùy vào dữ liệu
            'straight', 
            'traffic-light-red',
            'traffic-light_green', 
            'unknown'
        ]
        
        # TODO: Khởi tạo onnxruntime.InferenceSession hoặc tensorrt engine tại đây
        print(f"[CNNSignDetector] Đã khởi tạo lớp vỏ, sẵn sàng load ONNX từ {self.model_path}")

    def _infer_roi_onnx(self, roi):
        """
        Giả lập việc đẩy 1 ảnh ROI qua mô hình ONNX để phân loại.
        """
        # --- MÃ GIẢ LẬP ---
        # Ở đây bạn sẽ:
        # 1. cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # 2. Resize về kích thước model yêu cầu (vd: 128x128)
        # 3. Normalize & tạo batch tensor (1, C, H, W)
        # 4. output = self.sess.run(None, {self.input_name: tensor_data})[0]
        # 5. softmax(output) -> class_idx, confidence
        
        # Trả về nhãn unknown giả lập để không bị sập chương trình
        return "unknown", 0.0

    def detect(self, image):
        """
        Phát hiện ROI bằng màu sắc, sau đó phân loại bằng CNN.
        Trả về danh sách các biển báo/đèn giao thông được phát hiện.
        Format: [{"class": "left", "confidence": 0.95, "bbox": (x,y,w,h)}, ...]
        """
        results = []
        if image is None: return results

        # 1. Phát hiện Bounding Box bằng bộ lọc màu HSV
        detections, _ = self.color_detector.detect(image)
        
        # 2. Chạy CNN trên từng ROI
        for color_name, bboxes in detections.items():
            for (x, y, w, h) in bboxes:
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                # Gọi inference (ONNX/TensorRT)
                pred_class, conf_score = self._infer_roi_onnx(roi)
                
                # Loại bỏ kết quả nếu độ tự tin thấp
                if pred_class != "unknown" and conf_score >= self.threshold:
                    results.append({
                        "class": pred_class,
                        "confidence": conf_score,
                        "bbox": (x, y, w, h)
                    })

        return results

    def draw_debug(self, image, results):
        """Vẽ kết quả lên ảnh để debug."""
        output = image.copy()
        for res in results:
            x, y, w, h = res["bbox"]
            label = f'{res["class"]} ({res["confidence"]:.2f})'
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output, label, (x, max(15, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return output
