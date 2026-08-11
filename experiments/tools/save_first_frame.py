import cv2

video_path = r"e:\robot-jeston\logs\logs\v3_20260810_185619.avi"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if ret:
    cv2.imwrite(r"e:\robot-jeston\experiments\first_frame.jpg", frame)
    print("Saved first_frame.jpg")
