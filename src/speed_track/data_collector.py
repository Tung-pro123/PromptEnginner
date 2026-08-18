#!/usr/bin/env python3
"""
Data Collector Script cho Mô hình AI Segmentation
Thu thập ảnh từ Camera ROS và lưu xuống thư mục dataset/images.

Cách dùng:
1. Chạy script này trên Jetson: `python3 data_collector.py`
2. Dùng tay cầm (Gamepad) hoặc Teleop để lái xe chạy quanh sa bàn.
3. Script sẽ tự động chụp 5 ảnh mỗi giây (5 FPS) để tạo Dataset.
4. Bấm Ctrl+C để dừng.
"""

import os
import cv2
import numpy as np
import rospy
import time
from sensor_msgs.msg import Image

class AIDataCollector:
    def __init__(self):
        rospy.init_node('ai_data_collector', anonymous=True)
        
        # Tạo thư mục lưu ảnh
        self.dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'images')
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.camera_topic = '/csi_cam_0/image_raw'
        self.latest_image = None
        self.image_count = 0
        
        rospy.Subscriber(self.camera_topic, Image, self._cam_cb, queue_size=1)
        rospy.loginfo(f"Đã bắt đầu kết nối tới camera: {self.camera_topic}")
        rospy.loginfo(f"Ảnh sẽ được lưu tại: {self.dataset_dir}")

    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                self.latest_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
        except Exception as e:
            rospy.logerr_throttle(5, f"Camera error: {e}")

    def run(self):
        rospy.loginfo("Đang chờ ảnh từ Camera...")
        while self.latest_image is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        rospy.loginfo("Đã nhận được ảnh! Bắt đầu ghi hình tự động (5 ảnh/giây)...")
        rospy.loginfo("HÃY ĐIỀU KHIỂN XE CHẠY QUANH SA BÀN (CÓ VÀ KHÔNG CÓ VẬT CẢN). Bấm Ctrl+C để dừng.")
        
        rate = rospy.Rate(5) # 5 FPS là đủ tốt cho Dataset, tránh bị trùng lặp ảnh quá nhiều
        
        try:
            while not rospy.is_shutdown():
                if self.latest_image is not None:
                    # Lấy timestamp làm tên file để đảm bảo duy nhất
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    ms = int((time.time() % 1) * 1000)
                    filename = f"img_{ts}_{ms:03d}.jpg"
                    filepath = os.path.join(self.dataset_dir, filename)
                    
                    # Lưu ảnh gốc (640x480)
                    cv2.imwrite(filepath, self.latest_image)
                    self.image_count += 1
                    
                    # Print thống kê mỗi 10 ảnh
                    if self.image_count % 10 == 0:
                        rospy.loginfo(f"Đã lưu {self.image_count} ảnh vào dataset.")
                        
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            rospy.loginfo(f"Hoàn tất! Tổng cộng đã chụp: {self.image_count} ảnh.")

if __name__ == '__main__':
    try:
        collector = AIDataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass
