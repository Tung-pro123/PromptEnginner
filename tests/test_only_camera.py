#!/usr/bin/env python3
"""
Kiểm tra RIÊNG LẺ: Camera CSI (đọc qua OpenCV GStreamer và ROS)
Chạy trên Jetson:
    python3 tests/test_only_camera.py
"""
import sys
# Sắp xếp lại sys.path để ưu tiên các thư viện Python 3 trước, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import os
import cv2
import time

# Thêm thư mục gốc chứa src vào path để import nếu cần
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_next_image_path(prefix="frame"):
    """
    Tạo thư mục 'captured_images' ở gốc dự án và trả về đường dẫn file tiếp theo dạng prefix_n.jpg
    """
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(workspace_root, "captured_images")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"[INFO] Đã tạo thư mục lưu ảnh: {save_dir}")
        
    existing_files = os.listdir(save_dir)
    max_num = 0
    for file in existing_files:
        if file.startswith(prefix + "_") and file.endswith(".jpg"):
            try:
                # Trích xuất số thứ tự từ tên file (ví dụ: 'frame_direct_3.jpg' -> 3)
                parts = file.split("_")
                num_str = parts[-1].split(".")[0]
                num = int(num_str)
                if num > max_num:
                    max_num = num
            except (ValueError, IndexError):
                pass
                
    next_num = max_num + 1
    filename = f"{prefix}_{next_num}.jpg"
    return os.path.join(save_dir, filename), filename

def test_gstreamer_camera():
    print("--- 1. Kiểm tra mở Camera trực tiếp bằng GStreamer ---")
    pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)300, height=(int)300, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[OK] Kết nối trực tiếp GStreamer THÀNH CÔNG.")
        ret, frame = cap.read()
        if ret:
            print(f"[OK] Đã chụp được khung hình. Kích thước: {frame.shape}")
            filepath, filename = get_next_image_path("frame_direct")
            cv2.imwrite(filepath, frame)
            print(f"-> Đã lưu ảnh vào: 'captured_images/{filename}'")
        cap.release()
    else:
        print("[ERROR] Không mở được Camera qua GStreamer.")

def test_ros_camera():
    print("\n--- 2. Kiểm tra nhận ảnh qua ROS Topic ---")
    import rospy
    from sensor_msgs.msg import Image
    
    rospy.init_node('test_only_camera_node', anonymous=True)
    frame_received = []
    
    def cb(msg):
        frame_received.append(msg)
        print(f"[OK] Đã nhận được ảnh từ ROS! Độ phân giải: {msg.width}x{msg.height}")
        
        # Chuyển đổi ảnh ROS sang OpenCV để lưu
        try:
            import numpy as np
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                frame = img.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img_rgb = img.reshape((msg.height, msg.width, 3))
                frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                frame = img.reshape((msg.height, msg.width))
            else:
                frame = None
                
            if frame is not None:
                filepath, filename = get_next_image_path("frame_ros")
                cv2.imwrite(filepath, frame)
                print(f"-> Đã lưu ảnh ROS thành công vào: 'captured_images/{filename}'")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể chuyển đổi và lưu ảnh ROS: {e}")
            
        rospy.signal_shutdown("Đã nhận được ảnh")

    rospy.Subscriber('/csi_cam_0/image_raw', Image, cb)
    rospy.Subscriber('/camera/image_raw', Image, cb)
    
    print("Đang chờ nhận 1 ảnh từ ROS (chờ tối đa 5s)...")
    start = time.time()
    while not rospy.is_shutdown() and time.time() - start < 5.0:
        rospy.sleep(0.1)
        
    if not frame_received:
        print("[ERROR] Không nhận được ảnh từ ROS. Đảm bảo roslaunch camera đã chạy.")

if __name__ == '__main__':
    test_gstreamer_camera()
    try:
        test_ros_camera()
    except Exception as e:
        print(f"Không test được ROS: {e}")
