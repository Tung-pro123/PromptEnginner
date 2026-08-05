import cv2
import numpy as np

class TrafficLightClassifier:
    def __init__(self):
        # Define HSV ranges for Red and Green colors
        # Red has two ranges due to Hue wrapping around 0/180
        self.red_lower1 = np.array([0, 70, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 70, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([40, 70, 70])
        self.green_upper = np.array([90, 255, 255])
        
        # Minimum pixel threshold to qualify a detection
        self.min_pixel_ratio = 0.05  # 5% of crop area

    def crop_light_region(self, frame, bbox):
        """
        Crops the bounding box of the traffic light from the frame.
        bbox format: (x, y, w, h)
        """
        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = bbox
        
        # Ensure bounding box is within frame boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return frame[y1:y2, x1:x2]

    def classify_color(self, crop):
        """
        Classifies the traffic light color using HSV color thresholding.
        Returns: "RED", "GREEN", or "UNKNOWN"
        """
        if crop is None or crop.size == 0:
            return "UNKNOWN"
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Create masks for Red and Green
        mask_red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_green = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Count the number of active pixels
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        
        total_pixels = crop.shape[0] * crop.shape[1]
        red_ratio = float(red_pixels) / total_pixels
        green_ratio = float(green_pixels) / total_pixels
        
        # Classification decision logic
        if red_ratio > green_ratio and red_ratio >= self.min_pixel_ratio:
            return "RED"
        elif green_ratio > red_ratio and green_ratio >= self.min_pixel_ratio:
            return "GREEN"
        else:
            return "UNKNOWN"
