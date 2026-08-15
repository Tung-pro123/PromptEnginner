#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

class BEVCalibrator:
    def __init__(self):
        rospy.init_node('bev_calibrator', anonymous=True)
        self.latest_image = None
        self.points = []
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.cam_cb)
        cv2.namedWindow('Select 4 Points')
        cv2.setMouseCallback('Select 4 Points', self.click_event)

    def cam_cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")

    def run(self):
        print("Đang chạy BEV Calibrator.")
        print("Hãy click 4 điểm trên mặt đường tạo thành một hình thang (Trapezoid).")
        print("Thứ tự click: Trái-Dưới, Phải-Dưới, Phải-Trên, Trái-Trên.")
        print("Nhấn 'q' để thoát, 'r' để chọn lại điểm.")
        
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                img_show = self.latest_image.copy()
                
                # Vẽ các điểm đã click
                for i, p in enumerate(self.points):
                    cv2.circle(img_show, p, 5, (0, 255, 0), -1)
                    cv2.putText(img_show, str(i+1), (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    
                if len(self.points) == 4:
                    cv2.polylines(img_show, [np.array(self.points)], True, (255, 0, 0), 2)
                    
                    # Tính toán BEV
                    h, w = img_show.shape[:2]
                    src = np.float32(self.points)
                    # Giả sử ta muốn map thành hcn ở giữa màn hình
                    dst = np.float32([
                        [w*0.25, h],
                        [w*0.75, h],
                        [w*0.75, 0],
                        [w*0.25, 0]
                    ])
                    matrix = cv2.getPerspectiveTransform(src, dst)
                    bev_img = cv2.warpPerspective(self.latest_image, matrix, (w, h))
                    cv2.imshow('Bird Eye View', bev_img)
                
                cv2.imshow('Select 4 Points', img_show)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if len(self.points) == 4:
                        print("\n=== THÔNG SỐ CỦA BẠN ===")
                        print(f"BEV_SRC_PTS = np.float32({self.points})")
                        print("Sao chép mảng này vào config của thuật toán.")
                    break
                elif key == ord('r'):
                    self.points = []
                    print("Đã reset điểm. Vui lòng click lại.")
            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    BEVCalibrator().run()
