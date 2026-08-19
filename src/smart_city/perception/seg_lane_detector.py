#!/usr/bin/env python3
"""
YOLO Segmentation Lane & Crosswalk Detector
Nhận diện đường biên và vạch qua đường bằng Instance Segmentation.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Any
from ultralytics import YOLO

@dataclass
class SegLaneResult:
    center_x: Optional[int] = None
    left_x: Optional[int] = None
    right_x: Optional[int] = None
    mode: str = 'LOST'  # 'BOTH', 'LEFT_ONLY', 'RIGHT_ONLY', 'LOST'
    has_crosswalk: bool = False
    raw_results: Any = None  # Để debug

class YoloSegDetector:
    def __init__(self, model_path, config):
        self.cfg = config
        print(f"Loading YOLO-seg model from {model_path}...")
        self.model = YOLO(model_path, task='segment')
        self.last_crosswalk_time = 0.0

    def detect(self, image) -> SegLaneResult:
        if image is None:
            return SegLaneResult()
            
        # Chạy inference với ngưỡng tin cậy conf=0.5 để loại bỏ các nhiễu (mask rác) ở giao lộ
        results = self.model(image, verbose=False, imgsz=320, conf=0.5)[0]
        
        left_mask = None
        right_mask = None
        crosswalk_mask = None
        
        if results.masks is not None:
            for i, cls in enumerate(results.boxes.cls):
                cls_id = int(cls.item())
                # Mask tensor, reshape lại bằng OpenCV cho mượt
                mask = results.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # 0: Crosswalk, 1: Left, 2: Right (Dựa theo data.yaml)
                if cls_id == 1:
                    left_mask = mask if left_mask is None else np.maximum(left_mask, mask)
                elif cls_id == 2:
                    right_mask = mask if right_mask is None else np.maximum(right_mask, mask)
                elif cls_id == 0:
                    crosswalk_mask = mask if crosswalk_mask is None else np.maximum(crosswalk_mask, mask)
        
        # Trích xuất trọng tâm của mép dưới
        left_x = self._get_bottom_centroid_x(left_mask)
        right_x = self._get_bottom_centroid_x(right_mask)
        
        center_x = None
        mode = 'LOST'

        if left_x is not None and right_x is not None:
            center_x = (left_x + right_x) // 2
            mode = 'BOTH'
        elif left_x is not None:
            center_x = left_x + self.cfg.lane_half_width_px
            mode = 'LEFT_ONLY'
        elif right_x is not None:
            center_x = right_x - self.cfg.lane_half_width_px
            mode = 'RIGHT_ONLY'

        # Nhận diện Crosswalk
        has_crosswalk = False
        if crosswalk_mask is not None:
            # Chỉ xét Crosswalk khi nó nằm gần xe (nửa dưới ảnh)
            h = image.shape[0]
            bottom_crosswalk = crosswalk_mask[int(h*0.5):, :]
            crosswalk_area = np.sum(bottom_crosswalk > 0.5)
            
            # Crosswalk đủ lớn thì mới tính là giao lộ
            if crosswalk_area > self.cfg.min_contour_area * 5:
                has_crosswalk = True
                self.last_crosswalk_time = time.time()
                
        return SegLaneResult(
            center_x=center_x, 
            left_x=left_x, 
            right_x=right_x, 
            mode=mode, 
            has_crosswalk=has_crosswalk,
            raw_results=results
        )

    def _get_bottom_centroid_x(self, mask):
        if mask is None:
            return None
            
        h, w = mask.shape
        # Lấy 40% phần dưới cùng của ảnh để tính toán (gần đầu xe nhất)
        roi_top = int(h * 0.6)
        bottom_half = mask[roi_top:h, :]
        
        contours, _ = cv2.findContours((bottom_half > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return None
            
        M = cv2.moments(largest)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"])
        return None

    def draw_debug(self, image, result: SegLaneResult):
        """Vẽ thông tin debug lên ảnh."""
        if image is None or result.raw_results is None:
            return image

        # Vẽ segmentation masks mặc định của Ultralytics
        dbg = result.raw_results.plot()
        
        # Vẽ các điểm tính toán PID
        h = dbg.shape[0]
        y_draw = int(h * 0.8) # Điểm vẽ center
        
        if result.left_x is not None:
            cv2.circle(dbg, (result.left_x, y_draw), 8, (255, 0, 0), -1) # Blue
        if result.right_x is not None:
            cv2.circle(dbg, (result.right_x, y_draw), 8, (0, 255, 0), -1) # Green
        if result.center_x is not None:
            cv2.line(dbg, (result.center_x, y_draw - 20), (result.center_x, y_draw + 20), (0, 0, 255), 3) # Red center

        cv2.putText(dbg, f"Mode: {result.mode}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if result.has_crosswalk:
            cv2.putText(dbg, "CROSSWALK DETECTED!", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return dbg
