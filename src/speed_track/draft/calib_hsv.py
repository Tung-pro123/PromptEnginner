#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

class HSVCalibrator:
    def __init__(self):
        rospy.init_node('hsv_calibrator', anonymous=True)
        self.latest_image = None
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.cam_cb)
        
        cv2.namedWindow('Calibration')
        # Tạo 2 dải màu vì màu đỏ nằm ở 2 đầu của phổ Hue (0-10 và 160-180)
        cv2.createTrackbar('H1 Min', 'Calibration', 0, 180, self.nothing)
        cv2.createTrackbar('S1 Min', 'Calibration', 80, 255, self.nothing)
        cv2.createTrackbar('V1 Min', 'Calibration', 80, 255, self.nothing)
        cv2.createTrackbar('H1 Max', 'Calibration', 18, 180, self.nothing)
        
        cv2.createTrackbar('H2 Min', 'Calibration', 155, 180, self.nothing)
        cv2.createTrackbar('H2 Max', 'Calibration', 180, 180, self.nothing)

    def nothing(self, x): pass

    def cam_cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img

    def run(self):
        print("Đang chạy HSV Calibrator. Kéo thanh trượt để chỉnh. Nhấn 'q' để thoát.")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                # 1. Tiền xử lý CLAHE chống nhiễu sáng (Giống hệt plan)
                lab = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                lab = cv2.merge((cl,a,b))
                bgr_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                hsv = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2HSV)
                
                # 2. Đọc giá trị thanh trượt
                h1_min = cv2.getTrackbarPos('H1 Min', 'Calibration')
                s1_min = cv2.getTrackbarPos('S1 Min', 'Calibration')
                v1_min = cv2.getTrackbarPos('V1 Min', 'Calibration')
                h1_max = cv2.getTrackbarPos('H1 Max', 'Calibration')
                
                h2_min = cv2.getTrackbarPos('H2 Min', 'Calibration')
                h2_max = cv2.getTrackbarPos('H2 Max', 'Calibration')
                
                # 3. Tạo Mask
                lower1 = np.array([h1_min, s1_min, v1_min])
                upper1 = np.array([h1_max, 255, 255])
                lower2 = np.array([h2_min, s1_min, v1_min])
                upper2 = np.array([h2_max, 255, 255])
                
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
                
                res = cv2.bitwise_and(self.latest_image, self.latest_image, mask=mask)
                
                # 4. Hiển thị
                combined = cv2.hconcat([cv2.resize(self.latest_image, (320, 240)), cv2.resize(res, (320, 240))])
                cv2.imshow('Left: Gốc - Right: Filtered', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n=== THÔNG SỐ CỦA BẠN ===")
                    print(f"RED_LOWER_1 = np.array([{h1_min}, {s1_min}, {v1_min}])")
                    print(f"RED_UPPER_1 = np.array([{h1_max}, 255, 255])")
                    print(f"RED_LOWER_2 = np.array([{h2_min}, {s1_min}, {v1_min}])")
                    print(f"RED_UPPER_2 = np.array([{h2_max}, 255, 255])")
                    break
            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    HSVCalibrator().run()
