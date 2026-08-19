import cv2
import numpy as np
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")
    sys.exit(1)

class ImageSegmenter:
    def __init__(self, model_path="best.pt"):
        """
        Khởi tạo class phân đoạn ảnh.
        :param model_path: Đường dẫn tới model YOLOv8 segmentation (đuôi .pt)
        """
        print(f"[INFO] Đang tải mô hình YOLO từ: {model_path}...")
        try:
            self.model = YOLO(model_path)
            # Lấy danh sách các class ID được phép vẽ (bỏ qua 'crosswalk')
            self.allowed_classes = [k for k, v in self.model.names.items() if 'crosswalk' not in v.lower()]
            print(f"[INFO] Các class sẽ được nhận diện: {[self.model.names[k] for k in self.allowed_classes]}")
        except Exception as e:
            print(f"[ERROR] Không thể load model. Chi tiết: {e}")
            sys.exit(1)
            
        # Kích thước màn hình mặc định
        self.W = 640
        self.H = 480

    def process_frame(self, frame):
        """
        Nhận vào 1 frame (ảnh numpy từ OpenCV), phân đoạn và vẽ debug.
        :return: debug_image
        """
        # Đảm bảo kích thước ảnh chuẩn
        frame = cv2.resize(frame, (self.W, self.H))
        
        # Chạy AI dự đoán trên frame, lọc bỏ crosswalk thông qua tham số classes
        # imgsz=320 để quá trình infer nhanh hơn (tối ưu cho Jetson/thiết bị nhúng)
        results = self.model.predict(
            source=frame, 
            imgsz=320, 
            verbose=False, 
            conf=0.1, 
            device='cpu',
            classes=self.allowed_classes
        )
        result = results[0]
        
        # Vẽ mask mà YOLO nhận diện được đè lên ảnh gốc (plot() trả về numpy frame)
        dbg_image = result.plot()

        if result.masks is None or len(result.masks.data) == 0:
            cv2.putText(dbg_image, "No Drivable Area Found", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return dbg_image


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Test YOLO Segmentation AI")
    parser.add_argument('--video', type=str, required=True, help="Đường dẫn đến file video")
    # Trỏ thẳng đến file model trong thư mục models của project gốc
    parser.add_argument('--model', type=str, default="../../models/segment_smart_city.pt", help="Đường dẫn đến file model (.pt)")
    args = parser.parse_args()

    model_file = args.model
    if not os.path.exists(model_file):
        print(f"[LỖI] Không tìm thấy file model: {model_file}")
        sys.exit(1)

    segmenter = ImageSegmenter(model_path=model_file)
    
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[LỖI] Không tìm thấy video: {video_path}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    print(f"\n[INFO] Đang phát video: {video_path}. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video hoặc lỗi đọc ảnh.")
            break
            
        debug_frame = segmenter.process_frame(frame)
        cv2.imshow("Segment Debug", debug_frame)
        
        # Delay 30ms để video phát ở tốc độ bình thường (~30fps)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
