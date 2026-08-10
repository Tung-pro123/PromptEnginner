import cv2
import numpy as np

def analyze_video():
    video_path = r"e:\robot-jeston\logs\logs\v3_20260810_185619.avi"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video")
        return
        
    H, W, _ = frame.shape
    print(f"Video resolution: {W}x{H}")
    
    # Tìm các pixel màu vàng (0, 255, 255) trong BGR
    # BGR for yellow is (0, 255, 255)
    yellow_mask = cv2.inRange(frame, (0, 200, 200), (50, 255, 255))
    yellow_coords = cv2.findNonZero(yellow_mask)
    if yellow_coords is not None:
        x, y, w, h = cv2.boundingRect(yellow_coords)
        print(f"Yellow bounding box: x={x}, y={y}, w={w}, h={h}")
    else:
        print("Yellow not found")
        
    # Tìm pixel màu xanh blue (255, 0, 0) trong BGR
    blue_mask = cv2.inRange(frame, (200, 0, 0), (255, 50, 50))
    blue_coords = cv2.findNonZero(blue_mask)
    if blue_coords is not None:
        x, y, w, h = cv2.boundingRect(blue_coords)
        print(f"Blue bounding box: x={x}, y={y}, w={w}, h={h}")
    else:
        print("Blue not found")
        
if __name__ == "__main__":
    analyze_video()
