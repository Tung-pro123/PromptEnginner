#!/usr/bin/env python3
import cv2
import numpy as np
from src.smart_city.perception.dual_lane_detector import DualLaneDetector, DualLaneResult

class YoloSegModel:
    """Lớp vỏ (Wrapper) cho model Yolo Segmentation (sẽ dùng ONNX/TensorRT sau)"""
    def __init__(self, model_path="models/yolo_seg_lane.onnx"):
        self.model_path = model_path
        # TODO: Khởi tạo onnxruntime.InferenceSession hoặc tensorrt engine tại đây
        print(f"[YoloSegModel] Đã khởi tạo vỏ mô hình sẵn sàng load {self.model_path}")

    def infer(self, image_roi):
        """
        Chạy inference Yolo Seg.
        Input: Ảnh ROI (BGR)
        Output: Ảnh nhị phân (mask) có cùng kích thước, 255 ở những pixel là làn đường.
        """
        # --- MÃ GIẢ LẬP (Dummy) ---
        # Thực tế ở đây sẽ là:
        # 1. Preprocess (resize, normalize, transpose thành NCHW)
        # 2. Run ONNX: outputs = self.sess.run(None, {self.input_name: tensor})
        # 3. Postprocess: Trích xuất mask từ output (thresholding)
        
        mask = np.zeros(image_roi.shape[:2], dtype=np.uint8)
        # Trả về mask rỗng tạm thời
        return mask

class SegDualLaneDetector(DualLaneDetector):
    """
    Kế thừa DualLaneDetector, nhưng thay vì dùng HSV Color Mask (cv2.inRange),
    ta dùng mask sinh ra từ mạng Yolo Segmentation.
    """
    def __init__(self, config, model_path="models/yolo_seg_lane.onnx"):
        super().__init__(config)
        self.seg_model = YoloSegModel(model_path)

    def _create_line_mask(self, roi):
        """
        Ghi đè (Override) phương thức tạo mask của class cha.
        Sử dụng Yolo Seg thay vì HSV.
        """
        # Gọi model Segmentation để lấy mask của hai làn
        seg_mask = self.seg_model.infer(roi)
        
        # Vẫn có thể áp dụng Morphological operations từ config của cha nếu muốn làm sạch mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size))
        if self.cfg.morph_open_iter > 0:
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel, iterations=self.cfg.morph_open_iter)
        if self.cfg.morph_close_iter > 0:
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.morph_close_iter)
            
        return seg_mask
