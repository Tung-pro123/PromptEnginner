import cv2
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import các modules có sẵn trong dự án
from sign_color_detector import SignColorDetector
from train_cnn import SimpleCNN

class SignRecognizerCNN:
    def __init__(self, model_path=r'e:\robot-jeston\models\best_cnn_imbalanced_signs.pth'):
        self.color_detector = SignColorDetector()
        
        # 1. Định nghĩa Classes (theo thứ tự Alphabet mà ImageFolder của PyTorch tạo ra lúc train)
        self.classes = [
            'forbidden', 
            'left', 
            'straight', 
            'traffic-light-red',
            'traffic-light_green', 
            'unknown'
        ]
        self.num_classes = len(self.classes)
        
        # 2. Khởi tạo thiết bị & Mô hình
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Đang nạp mô hình CNN lên thiết bị: {self.device}...")
        
        self.model = SimpleCNN(num_classes=self.num_classes).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # Chuyển sang chế độ inference (tắt Dropout)
            print("✅ Đã nạp thành công trọng số mạng CNN!")
        else:
            print(f"❌ CẢNH BÁO: Không tìm thấy file model tại {model_path}. Hãy chạy train_cnn.py trước.")
            
        # 3. Tiền xử lý ảnh (Transform) y hệt như lúc Train
        self.img_size = 128
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def recognize(self, image, debug=True):
        # Phát hiện Bounding Box bằng bộ lọc màu HSV cực nhẹ của OpenCV
        detections, masks = self.color_detector.detect(image)
        
        output_image = image.copy()
        debug_crops = []
        
        draw_colors = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0)
        }
        
        for color_name, bboxes in detections.items():
            color_bgr = draw_colors.get(color_name, (255, 255, 255))
            
            for (x, y, w, h) in bboxes:
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                # THEO YÊU CẦU: Bỏ qua hoàn toàn bộ lọc Hình học (Circularity/Purity).
                # Nhường toàn bộ trách nhiệm phân loại rác/nhiễu cho mạng CNN qua nhãn 'unknown'.
                
                # 1. Chuyển đổi BGR (OpenCV) -> RGB (PIL)
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                
                # 2. Áp dụng Tensor Transform & Thêm chiều Batch (unsqueeze)
                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # 3. Chạy Inference qua mô hình CNN
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)
                    
                    pred_class = self.classes[predicted_idx.item()]
                    conf_score = confidence.item()
                
                # YÊU CẦU: Ngưỡng Confidence cực kỳ khắt khe (90%)
                # Bất kỳ dự đoán biển báo nào (không phải unknown) mà độ tự tin < 90% 
                # đều sẽ bị giáng cấp xuống thành 'unknown' (rác/nhiễu).
                if pred_class != "unknown" and conf_score < 0.90:
                    label = "unknown"
                else:
                    label = pred_class
                
                # KHÔNG vẽ hoặc đưa vào cửa sổ Debug nếu kết quả là rác (unknown)
                if label == "unknown":
                    continue
                    
                # Hiển thị
                text = f"{label} ({conf_score*100:.0f}%)"
                
                # 4. Vẽ trực quan lên ảnh
                cv2.rectangle(output_image, (x, y), (x+w, y+h), color_bgr, 2)
                
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_image, (x, y - text_h - 10), (x + text_w, y), color_bgr, -1)
                cv2.putText(output_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Lưu thông tin debug
                if debug:
                    roi_resized = cv2.resize(roi, (128, 128))
                    debug_crops.append((text, roi_resized))
                    
        return output_image, debug_crops



