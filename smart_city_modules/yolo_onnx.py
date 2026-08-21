import cv2
import numpy as np
import os

class CentroidTracker:
    """Tracker siêu nhẹ dựa trên khoảng cách Euclidean để thay thế Ultralytics tracking"""
    def __init__(self, max_disappeared=5, max_distance=50):
        self.next_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, rects):
        if len(rects) == 0:
            for uid in list(self.disappeared.keys()):
                self.disappeared[uid] += 1
                if self.disappeared[uid] > self.max_disappeared:
                    del self.objects[uid]
                    del self.disappeared[uid]
            return []

        input_centroids = np.array([(int(x), int(y)) for (x, y, w, h, label) in rects])
        labels = [label for (x, y, w, h, label) in rects]

        if len(self.objects) == 0:
            results = []
            for i in range(len(input_centroids)):
                self.objects[self.next_id] = input_centroids[i]
                self.disappeared[self.next_id] = 0
                results.append({"label": labels[i], "x": input_centroids[i][0], "y": input_centroids[i][1], "id": self.next_id})
                self.next_id += 1
            return results

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Tính khoảng cách Euclidean
        D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        results = []

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            if D[row, col] > self.max_distance: continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            results.append({"label": labels[col], "x": input_centroids[col][0], "y": input_centroids[col][1], "id": object_id})
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])).difference(used_rows)
        unused_cols = set(range(D.shape[1])).difference(used_cols)

        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                del self.objects[obj_id]
                del self.disappeared[obj_id]

        for col in unused_cols:
            self.objects[self.next_id] = input_centroids[col]
            self.disappeared[self.next_id] = 0
            results.append({"label": labels[col], "x": input_centroids[col][0], "y": input_centroids[col][1], "id": self.next_id})
            self.next_id += 1

        return results

class YoloONNX:
    def __init__(self, onnx_path, class_names):
        try:
            import onnxruntime as ort
        except ImportError:
            print("Lỗi: Không tìm thấy thư viện onnxruntime. Hãy chạy: pip3 install onnxruntime hoặc onnxruntime-gpu trên Jetson")
            sys.exit(1)
            
        # Thử khởi tạo với CUDA/TensorRT nếu có, nếu không fallback về CPU
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        self.class_names = class_names
        self.tracker = CentroidTracker()

    def infer_and_track(self, frame, conf_threshold=0.25):
        h, w = frame.shape[:2]
        
        # Tiền xử lý ảnh giống cv2.dnn.blobFromImage
        # YOLOv8 chuẩn bị ảnh đầu vào: RGB, scale 1/255.0, resize 640x640, NCHW
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.expand_dims(img, axis=0) # CHW to NCHW
        
        # Chạy inference với ONNX Runtime
        preds = self.session.run(None, {self.input_name: img})[0] # Shape: (1, classes+4, 8400)

        preds = preds[0].T # -> (8400, classes+4)
        boxes, scores, class_ids = [], [], []

        for row in preds:
            cls_scores = row[4:]
            max_score = np.max(cls_scores)
            if max_score >= conf_threshold:
                class_id = np.argmax(cls_scores)
                # Scale box về lại kích thước ảnh gốc
                x_c, y_c, bw, bh = row[0:4]
                rx, ry = w / 640.0, h / 640.0
                boxes.append([int((x_c - bw/2) * rx), int((y_c - bh/2) * ry), int(bw * rx), int(bh * ry)])
                scores.append(float(max_score))
                class_ids.append(class_id)

        # Non-Maximum Suppression (Lọc box trùng) - cv2.dnn.NMSBoxes vẫn chạy tốt trên mọi bản OpenCV
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
        
        rects = []
        annotated_frame = frame.copy()
        
        if len(indices) > 0:
            # Flatten cẩn thận để tương thích đa phiên bản OpenCV
            if isinstance(indices, tuple):
                flattened_indices = list(indices)
            elif hasattr(indices, 'flatten'):
                flattened_indices = indices.flatten()
            else:
                flattened_indices = [idx[0] if isinstance(idx, (list, np.ndarray)) else idx for idx in indices]
                
            for i in flattened_indices:
                # Đảm bảo index hợp lệ
                i = int(i)
                if i >= len(boxes):
                    continue
                    
                bx, by, bw, bh = boxes[i]
                c_id = int(class_ids[i])
                
                # Tránh lỗi list index out of range nếu model output class id bất thường
                if c_id < len(self.class_names):
                    label = self.class_names[c_id]
                else:
                    label = f"Class_{c_id}"
                    
                center_x, center_y = bx + bw/2, by + bh/2
                rects.append((center_x, center_y, bw, bh, label))
                
                # Vẽ box lên frame
                cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Đưa qua Tracker để lấy ID
        detections = self.tracker.update(rects)
        
        # Loại các box ảo nằm trước Decision để logic lọc nhiễu tốt hơn (giống code cũ)
        if len(detections) > 0:
            decisions = [d for d in detections if d["label"] == "Decision"]
            if decisions:
                best_dec = max(decisions, key=lambda d: d["y"])
                y_dec = best_dec["y"]
                
                valid_detections = []
                for d in detections:
                    if d["label"] in ["Interact", "Corner"]:
                        if d["y"] <= y_dec + 15:
                            valid_detections.append(d)
                    else:
                        valid_detections.append(d)
                detections = valid_detections
                
        return detections, annotated_frame
