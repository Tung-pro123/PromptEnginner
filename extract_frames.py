import cv2
import os

video_path = r'd:\FPT_University\JetsonAIRacer\PromptEnginner\video\raw_camera.avi'
dataset_dir = r'd:\FPT_University\JetsonAIRacer\PromptEnginner\src\speed_track\dataset\raw_images_hq'
os.makedirs(dataset_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: fps = 30 # fallback

# We want to extract around 5 frames per second
frame_skip = int(fps / 5)
if frame_skip <= 0: frame_skip = 1

count = 0
saved = 0

print(f"Extracting frames from {video_path}...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    if count % frame_skip == 0:
        filename = f"vid_raw_camera_{saved:04d}.jpg"
        # Lưu bằng JPG với chất lượng cao nhất (100) để không bị nhòe
        cv2.imwrite(os.path.join(dataset_dir, filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        saved += 1
        
    count += 1

cap.release()
print(f"Done! Extracted {saved} frames to {dataset_dir}")
