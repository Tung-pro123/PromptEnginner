import cv2
import numpy as np
import argparse

def nothing(x): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error opening video")
        return

    cv2.namedWindow('HSV Tuning', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('H1_MIN', 'HSV Tuning', 0, 179, nothing)
    cv2.createTrackbar('H1_MAX', 'HSV Tuning', 20, 179, nothing)
    cv2.createTrackbar('H2_MIN', 'HSV Tuning', 155, 179, nothing)
    cv2.createTrackbar('H2_MAX', 'HSV Tuning', 179, 179, nothing)
    cv2.createTrackbar('S_MIN', 'HSV Tuning', 55, 255, nothing)
    cv2.createTrackbar('V_MIN', 'HSV Tuning', 70, 255, nothing)
    
    # CLAHE options
    cv2.createTrackbar('USE_CLAHE', 'HSV Tuning', 1, 1, nothing)

    ret, frame = cap.read()
    paused = True

    print("=== HSV TUNING TOOL ===")
    print("- Press SPACE to play/pause video")
    print("- Press 'q' to quit and save parameters")
    print("Adjust the sliders so the RED/ORANGE lines are purely WHITE on the rightmost mask.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        
        # Get trackbar positions
        h1_min = cv2.getTrackbarPos('H1_MIN', 'HSV Tuning')
        h1_max = cv2.getTrackbarPos('H1_MAX', 'HSV Tuning')
        h2_min = cv2.getTrackbarPos('H2_MIN', 'HSV Tuning')
        h2_max = cv2.getTrackbarPos('H2_MAX', 'HSV Tuning')
        s_min = cv2.getTrackbarPos('S_MIN', 'HSV Tuning')
        v_min = cv2.getTrackbarPos('V_MIN', 'HSV Tuning')
        use_clahe = cv2.getTrackbarPos('USE_CLAHE', 'HSV Tuning')

        # Apply CLAHE if enabled
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            process_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            process_frame = frame.copy()

        # HSV Thresholding
        hsv = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([h1_min, s_min, v_min]), np.array([h1_max, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([h2_min, s_min, v_min]), np.array([h2_max, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphology OPEN to remove noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Overlay for visualization
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Display: Original | Color Filtered | Binary Mask
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, "Color Filtered", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(mask_bgr, "Binary Mask (Tune this!)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        display = np.hstack((frame, result, mask_bgr))
        display = cv2.resize(display, (1280, 320))
        
        cv2.imshow('HSV Tuning', display)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n==================================")
    print("FINAL HSV PARAMETERS (COPY THIS TO config.py):")
    print(f"hsv_h1_min: int = {h1_min}")
    print(f"hsv_h1_max: int = {h1_max}")
    print(f"hsv_h2_min: int = {h2_min}")
    print(f"hsv_h2_max: int = {h2_max}")
    print(f"hsv_s_min: int = {s_min}")
    print(f"hsv_v_min: int = {v_min}")
    print(f"use_clahe: bool = {bool(use_clahe)}")
    print("==================================\n")

if __name__ == '__main__':
    main()
