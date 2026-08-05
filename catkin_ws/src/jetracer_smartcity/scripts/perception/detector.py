import os
import cv2
import numpy as np

# Try importing TensorRT and PyCUDA. If not available, enable mock fallback.
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    print("[WARNING] TensorRT or PyCUDA not available. detector.py will run in MOCK mode.")

from utils.image_utils import preprocess_image
from utils.config_loader import load_yaml

class YoloDetector:
    def __init__(self, config_path="d:/AI_Project/racing_promax/catkin_ws/src/jetracer_smartcity/config/model_config.yaml"):
        # Load configuration
        config = load_yaml(config_path)
        self.model_cfg = config["model"]
        self.engine_path = self.model_cfg["engine_path"]
        self.input_size = tuple(self.model_cfg["input_size"])
        self.conf_threshold = self.model_cfg["conf_threshold"]
        self.nms_threshold = self.model_cfg["nms_threshold"]
        self.classes = {int(k): v for k, v in self.model_cfg["classes"].items()}
        
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        if TRT_AVAILABLE:
            self.load_engine()
        else:
            self.mock_mode = True

    def load_engine(self):
        """
        Loads the TensorRT engine from disk and allocates CUDA buffers.
        """
        if not os.path.exists(self.engine_path):
            print(f"[WARNING] Engine file not found at: {self.engine_path}. Switching to MOCK mode.")
            self.mock_mode = True
            return

        self.mock_mode = False
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load the binary TRT engine
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Store binding info
            self.allocations.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def infer(self, frame):
        """
        Executes YOLO inference.
        Returns: list of detections [{'label': label_str, 'confidence': float, 'bbox': (x, y, w, h)}]
        """
        if frame is None or frame.size == 0:
            return []
            
        start_time = cv2.getTickCount()
        
        # 1. Preprocess
        batch_img, ratio, pad_details = preprocess_image(frame, self.input_size)
        
        if self.mock_mode:
            # Simulated inference for development environments
            detections = self._get_mock_detections(frame)
            latency_ms = ((cv2.getTickCount() - start_time) / cv2.getTickFrequency()) * 1000.0
            return detections, latency_ms

        # 2. Copy inputs to device, execute context asynchronously, and retrieve outputs
        # YOLOv5 input binding is usually index 0
        np.copyto(self.inputs[0]['host'], batch_img.ravel())
        
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'])
        self.context.execute_async_v2(bindings=self.allocations, stream_id=0)
        
        # Assuming single output binding
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # Reshape flat output array
        output_shape = self.outputs[0]['shape']
        raw_output = self.outputs[0]['host'].reshape(output_shape)
        
        # 3. Postprocess (NMS and scaling)
        detections = self.postprocess(raw_output[0], ratio, pad_details, frame.shape)
        
        latency_ms = ((cv2.getTickCount() - start_time) / cv2.getTickFrequency()) * 1000.0
        return detections, latency_ms

    def postprocess(self, output, ratio, pad_details, orig_shape):
        """
        Processes model output (NMS, confidence filtering, scaling boxes back to original shape).
        output shape: [num_anchors, 5 + num_classes] (e.g. [3000, 14])
        """
        # Filter detections by objectness confidence
        candidates = output[output[:, 4] > self.conf_threshold]
        if len(candidates) == 0:
            return []

        # Box coordinates: x_center, y_center, width, height -> x1, y1, x2, y2
        box_coords = candidates[:, :4]
        box_x1 = box_coords[:, 0] - box_coords[:, 2] / 2
        box_y1 = box_coords[:, 1] - box_coords[:, 3] / 2
        box_x2 = box_coords[:, 0] + box_coords[:, 2] / 2
        box_y2 = box_coords[:, 1] + box_coords[:, 3] / 2
        
        # Get class indices and confidences
        class_probs = candidates[:, 5:]
        class_confs = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        
        # Total scores = objectness * class probability
        scores = candidates[:, 4] * class_confs
        
        # Filter scores again
        valid_idx = scores > self.conf_threshold
        boxes = np.stack([box_x1, box_y1, box_x2, box_y2], axis=1)[valid_idx]
        scores = scores[valid_idx]
        class_ids = class_ids[valid_idx]
        
        if len(boxes) == 0:
            return []
            
        # Pure numpy NMS (Non-Maximum Suppression)
        keep = self._numpy_nms(boxes, scores, self.nms_threshold)
        
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        
        # Scale bounding boxes back to original image shape
        # pad_details: (dw, left, top)
        dw, left, top = pad_details
        rx, ry = ratio
        
        detections = []
        for i in range(len(boxes)):
            # Remove padding
            x1 = (boxes[i, 0] - left) / rx
            y1 = (boxes[i, 1] - top) / ry
            x2 = (boxes[i, 2] - left) / rx
            y2 = (boxes[i, 3] - top) / ry
            
            # Clip to image boundaries
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(orig_shape[1], int(round(x2)))
            y2 = min(orig_shape[0], int(round(y2)))
            
            w = x2 - x1
            h = y2 - y1
            
            label_id = int(class_ids[i])
            label_name = self.classes.get(label_id, "unknown")
            
            detections.append({
                "label": label_name,
                "confidence": float(scores[i]),
                "bbox": (x1, y1, w, h)
            })
            
        return detections

    def _numpy_nms(self, boxes, scores, threshold):
        """
        Pure numpy implementation of Non-Maximum Suppression (NMS).
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
            
        return keep

    def _get_mock_detections(self, frame):
        """
        Returns mock detections based on color/simple thresholds for testing.
        Used for verification when CUDA/TensorRT is not available.
        """
        # Look for red-ish or green-ish regions at the top/center to mock traffic light
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        detections = []
        # Mock detection coordinates if we detect a significant color block
        # For simplicity in mock tests, we check standard ranges:
        # Green mask
        mask_green = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([90, 255, 255]))
        green_cnt = cv2.countNonZero(mask_green)
        
        # Red mask
        mask_red1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_cnt = cv2.countNonZero(mask_red)
        
        if green_cnt > 200:
            detections.append({
                "label": "green_light",
                "confidence": 0.88,
                "bbox": (100, 50, 40, 80)
            })
        elif red_cnt > 200:
            detections.append({
                "label": "red_light",
                "confidence": 0.92,
                "bbox": (100, 50, 40, 80)
            })
            
        return detections
