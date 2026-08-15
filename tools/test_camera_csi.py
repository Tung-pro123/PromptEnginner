import cv2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def test_camera():
    print("Đang khởi động Camera CSI (Không dùng ROS)...")
    # Sử dụng pipeline GStreamer đặc thù của Jetson Nano
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print("✅ Camera đã bật thành công! Bấm 'q' để thoát.")
        cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("❌ Lỗi: Không thể mở camera. Hãy kiểm tra lại dây cắm hoặc khởi động lại xe.")

if __name__ == "__main__":
    test_camera()
