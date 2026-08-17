import cv2
import numpy as np
import os
import glob
from sign_color_detector import SignColorDetector

class SignRecognizer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.templates = {}
        self.detector = SignColorDetector()
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.load_database()

    def load_database(self):
        """Loads all templates and computes ORB features."""
        if not os.path.exists(self.db_path):
            print(f"Database path not found: {self.db_path}")
            return
            
        categories = [d for d in os.listdir(self.db_path) if os.path.isdir(os.path.join(self.db_path, d))]
        
        for category in categories:
            self.templates[category] = []
            cat_path = os.path.join(self.db_path, category)
            image_paths = glob.glob(os.path.join(cat_path, '**', '*.*'), recursive=True)
            
            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize template to a reasonable size for ORB (e.g. width=150)
                    h, w = img.shape[:2]
                    if w > 200:
                        scale = 200.0 / w
                        img = cv2.resize(img, (200, int(h * scale)))
                    
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    kp, des = self.orb.detectAndCompute(img_gray, None)
                    
                    # Store template info
                    self.templates[category].append({
                        "image": img,
                        "kp": kp,
                        "des": des
                    })
                    
        print(f"Loaded {sum(len(v) for v in self.templates.values())} templates.")

    def match_features(self, roi_bgr, color):
        """Matches a ROI against templates using ORB (scale/rotation invariant)."""
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Resize ROI slightly if it's too small for ORB to find features
        h, w = roi_gray.shape[:2]
        if w < 50 or h < 50:
            roi_gray = cv2.resize(roi_gray, (100, int(100 * h / max(1, w))))
            
        kp_roi, des_roi = self.orb.detectAndCompute(roi_gray, None)
        
        # Filter categories based on detected color
        allowed_categories = []
        if color == "red":
            allowed_categories = ["forbidden", "traffic-light"]
        elif color == "blue":
            allowed_categories = ["left", "right", "straight"]
        elif color == "green":
            allowed_categories = ["traffic-light"]
        else:
            allowed_categories = list(self.templates.keys())

        # If it's a traffic light, rely on shape rather than features (lights lack corners)
        if allowed_categories == ["traffic-light"] or color == "green":
            aspect_ratio = float(w) / max(1, h)
            if 0.6 < aspect_ratio < 1.4:
                return "traffic-light", 999.0 # Arbitrary high score for perfect shape
            else:
                return "unknown", 0.0

        if des_roi is None or len(des_roi) < 2:
            # Fallback for red color with no features (could be a solid red traffic light or plain circle)
            if color == "red":
                aspect_ratio = float(w) / max(1, h)
                if 0.6 < aspect_ratio < 1.4:
                    return "traffic-light", 50.0
            return "unknown", 0.0

        best_matches = -1
        best_class = "unknown"

        # Match against allowed categories
        for category in allowed_categories:
            if category not in self.templates or category == "traffic-light":
                # Traffic lights are handled by shape
                continue
                
            for template in self.templates[category]:
                des_template = template["des"]
                if des_template is None or len(des_template) < 2:
                    continue
                    
                matches = self.matcher.match(des_roi, des_template)
                # Filter good matches by distance
                good_matches = [m for m in matches if m.distance < 60]
                
                if len(good_matches) > best_matches:
                    best_matches = len(good_matches)
                    best_class = category
                    
        return best_class, float(best_matches)

    def recognize(self, image):
        """Detects signs and recognizes their types."""
        detections, _ = self.detector.detect(image)
        recognized_signs = []
        
        for color, bboxes in detections.items():
            for (x, y, w, h) in bboxes:
                # Add a small margin
                margin = int(min(w, h) * 0.1)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                    
                sign_class, score = self.match_features(roi, color)
                
                # Threshold for minimum good matches
                threshold = 5 if sign_class != "traffic-light" else 0
                
                if score > threshold:
                    recognized_signs.append({
                        "bbox": (x, y, w, h),
                        "color": color,
                        "class": sign_class,
                        "score": score
                    })
                    
        return recognized_signs

    def draw_results(self, image, recognized_signs):
        """Draws bounding boxes and labels on the image."""
        output = image.copy()
        
        for sign in recognized_signs:
            x, y, w, h = sign["bbox"]
            label = f'{sign["class"]} ({sign["score"]:.2f})'
            color_bgr = (0, 255, 0) # Green for text/box
            
            # Optional: adjust box color based on detected color
            if sign["color"] == "red":
                color_bgr = (0, 0, 255)
            elif sign["color"] == "blue":
                color_bgr = (255, 0, 0)
                
            cv2.rectangle(output, (x, y), (x+w, y+h), color_bgr, 2)
            cv2.putText(output, label, (x, max(15, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                        
        return output

if __name__ == "__main__":
    db_path = r'e:\robot-jeston\experiments\sign_database'
    test_folder = r'e:\robot-jeston\experiments\signs'
    
    recognizer = SignRecognizer(db_path)
    
    if os.path.exists(test_folder):
        image_paths = glob.glob(os.path.join(test_folder, '*.jpg')) + glob.glob(os.path.join(test_folder, '*.png'))
        print(f"Testing on {len(image_paths)} images...")
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Resize if too large
            height, width = img.shape[:2]
            max_height = 800
            if height > max_height:
                ratio = max_height / height
                img = cv2.resize(img, (int(width * ratio), max_height))
                
            results = recognizer.recognize(img)
            result_img = recognizer.draw_results(img, results)
            
            cv2.imshow('Sign Recognition', result_img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
