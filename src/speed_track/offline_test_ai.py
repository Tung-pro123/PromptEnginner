import cv2
import sys
import numpy as np
from ultralytics import YOLO

def test_offline_video(model_path, video_path):
    print(f"Đang tải mô hình từ: {model_path}...")
    model = YOLO(model_path)
    
    print(f"Đang mở video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Lỗi: Không thể mở video!")
        return

    W, H = 640, 480
    
    # Optional: Ghi ra file video kết quả
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('ai_result.mp4', fourcc, 30.0, (W, H))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Hết video.")
            break
            
        # Resize frame về kích thước chuẩn của hệ thống
        frame = cv2.resize(frame, (W, H))
        
        # Chạy AI Inference
        results = model.predict(source=frame, imgsz=320, verbose=False, conf=0.5, device='cpu')
        result = results[0]
        
        # Vẽ mask mặc định của YOLO để xem AI nhận diện thế nào
        dbg_image = result.plot() 
        
        if result.masks is not None:
            # Lấy mask
            # --- THUẬT TOÁN TÌM TÂM BẰNG LÁT CẮT NGANG (HORIZONTAL SLICE) ---
            # Khắc phục triệt để lỗi "tích lũy sai số do bị khuất làn đường" và "sai số do đường cong phình to"
            y_lookahead = int(H * 0.65) # Nhìn xa ở mức 65% màn hình
            row = mask[y_lookahead, :]
            road_pixels = np.where(row > 0)[0]
            
            if len(road_pixels) > 0:
                x_left = road_pixels[0]
                x_right = road_pixels[-1]
                
                # Tâm của đoạn đường nhìn thấy
                cX = int((x_left + x_right) / 2)
                cY = y_lookahead
                
                # BÙ TRỪ SAI SỐ KHI BỊ KHUẤT TẦM NHÌN (CAMERA CHỈ THẤY 1 PHẦN ĐƯỜNG)
                # Nếu mép trái bị khuất (chạm viền 0) nhưng mép phải vẫn thấy rõ -> True center nằm tuốt bên trái
                if x_left <= 5 and x_right < W - 10:
                    cX = cX - int(W * 0.15) # Ép tâm dịch mạnh sang trái để bù sai số
                # Nếu mép phải bị khuất (chạm viền W) nhưng mép trái vẫn thấy rõ -> True center nằm tuốt bên phải
                elif x_right >= W - 5 and x_left > 10:
                    cX = cX + int(W * 0.15) # Ép tâm dịch mạnh sang phải
                
                # Vẽ chấm hồng tại trọng tâm
                cv2.circle(dbg_image, (cX, cY), 10, (255, 0, 255), -1)
                
                # Vẽ tia bẻ lái từ xe (đáy giữa màn hình) tới trọng tâm
                cv2.line(dbg_image, (int(W/2), H), (cX, cY), (0, 255, 255), 3)
                
                # Tính góc lỗi
                error_x = cX - (W / 2)
                Kp = 0.005
                steer_angle = error_x * Kp
                steer_angle = max(-1.0, min(1.0, steer_angle))
                
                # Hiển thị thông số
                cv2.putText(dbg_image, f"Steer: {steer_angle:.2f}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Offline AI Test", dbg_image)
        # out.write(dbg_image)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_file = r"runs\segment\jetson_track_seg-2\weights\best.pt"
    video_file = r"..\..\video\raw_camera.avi"
    test_offline_video(model_file, video_file)
