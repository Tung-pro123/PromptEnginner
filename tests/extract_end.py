import cv2
import os

def main():
    video_path = r"c:\Users\Lenovo\OneDrive\Documents\Jeston\PromptEnginner\speed_track_run.avi"
    output_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\6c05630f-e2f4-4ed3-bf22-1ce0e508e31c"
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract the last 60 frames (approx 3 seconds at 20 FPS)
    start_frame = max(0, frame_count - 60)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Extracting frames from {start_frame} to {frame_count}...")
    
    frame_idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        out_name = f"media__end_frame_{frame_idx:04d}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, frame)
        frame_idx += 1
        
    cap.release()
    print("Done extracting end frames.")

if __name__ == "__main__":
    main()
