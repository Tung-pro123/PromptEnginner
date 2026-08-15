#!/usr/bin/env python3
"""
Kiểm tra RIÊNG LẺ: Cảm biến LiDAR (RPLIDAR) - Chế độ theo dõi liên tục
Hiển thị bảng điều khiển khoảng cách vật cản theo thời gian thực.

Chạy trên Jetson:
    python3 tests/test_only_lidar.py

Nhấn Ctrl+C để dừng.
"""
import sys
# Sắp xếp lại sys.path để ưu tiên các thư viện Python 3 trước, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import os
import time
import math

try:
    import rospy
    from sensor_msgs.msg import LaserScan
    HAS_ROS = True
except ImportError:
    rospy = None
    HAS_ROS = False

# Thêm thư mục gốc chứa src vào path để import nếu cần
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LidarMonitor:
    def __init__(self):
        rospy.init_node('lidar_monitor_node', anonymous=True)
        self.scan_data = None
        self.scan_count = 0
        rospy.Subscriber('/scan', LaserScan, self.callback)
        print("Đang chờ kết nối LiDAR...")

    def callback(self, msg):
        self.scan_data = msg
        self.scan_count += 1

    def get_zone_distance(self, msg, center_deg, half_range_deg):
        """Lấy khoảng cách gần nhất trong một vùng góc nhất định."""
        distances = []
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            # Bù 180° vì LiDAR lắp ngược hướng trên xe
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            low = center_deg - half_range_deg
            high = center_deg + half_range_deg
            
            if low <= angle_deg <= high:
                if msg.range_min < dist < msg.range_max:
                    distances.append(dist)
        
        if distances:
            return min(distances), len(distances)
        return float('inf'), 0

    def make_bar(self, dist, max_dist=3.0, bar_width=20):
        """Tạo thanh biểu đồ ASCII cho khoảng cách."""
        if dist == float('inf'):
            return "[" + " " * bar_width + "] ---"
        ratio = min(dist / max_dist, 1.0)
        filled = int(ratio * bar_width)
        if dist < 0.30:
            marker = "!"  # Nguy hiểm
        elif dist < 0.60:
            marker = "#"  # Cảnh báo
        else:
            marker = "="  # An toàn
        bar = "[" + marker * filled + " " * (bar_width - filled) + "]"
        return f"{bar} {dist:.2f}m"

    def get_status_icon(self, dist):
        """Trả về biểu tượng trạng thái theo khoảng cách."""
        if dist == float('inf'):
            return "     "
        elif dist < 0.30:
            return " <!> "  # Rất gần - Nguy hiểm
        elif dist < 0.50:
            return " <W> "  # Cảnh báo
        elif dist < 1.00:
            return " [i] "  # Phát hiện
        else:
            return "  .  "  # Xa

    def run(self):
        rate = rospy.Rate(5)  # Cập nhật 5 lần/giây
        
        # Chờ nhận dữ liệu đầu tiên
        timeout = time.time() + 8.0
        while self.scan_data is None and not rospy.is_shutdown():
            if time.time() > timeout:
                print("[ERROR] Không nhận được dữ liệu từ LiDAR sau 8 giây.")
                print("Hướng dẫn: Chạy 'roslaunch jetracer lidar.launch' ở terminal khác.")
                return
            rospy.sleep(0.1)
        
        print("\n" + "=" * 60)
        print("  BANG DIEU KHIEN LiDAR THOI GIAN THUC")
        print("  Nhan Ctrl+C de dung")
        print("=" * 60)
        
        while not rospy.is_shutdown():
            msg = self.scan_data
            if msg is None:
                rate.sleep()
                continue
            
            # Đo khoảng cách ở 5 vùng xung quanh xe
            front_dist, front_pts   = self.get_zone_distance(msg, 0.0, 15.0)
            left_dist, left_pts     = self.get_zone_distance(msg, 90.0, 15.0)
            right_dist, right_pts   = self.get_zone_distance(msg, -90.0, 15.0)
            back_dist, back_pts     = self.get_zone_distance(msg, 180.0, 15.0)
            fl_dist, fl_pts         = self.get_zone_distance(msg, 45.0, 15.0)
            fr_dist, fr_pts         = self.get_zone_distance(msg, -45.0, 15.0)
            
            # Vật cản gần nhất toàn cục
            all_valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
            global_min = min(all_valid) if all_valid else float('inf')
            
            # Xóa màn hình và vẽ lại bảng điều khiển
            os.system('clear')
            
            print("=" * 60)
            print("  BANG DIEU KHIEN LiDAR THOI GIAN THUC")
            print(f"  Lan quet thu: {self.scan_count} | So diem/vong: {len(msg.ranges)}")
            print("=" * 60)
            
            # Hiển thị bản đồ vùng xung quanh xe (dạng chữ)
            print()
            print(f"              TRUOC ({front_pts} diem)")
            print(f"         {self.make_bar(front_dist)}")
            print(f"   {self.get_status_icon(fl_dist)} FL               FR {self.get_status_icon(fr_dist)}")
            print(f"   {fl_dist:.2f}m                {fr_dist:.2f}m")
            print()
            print(f"   TRAI                         PHAI")
            print(f"   {self.make_bar(left_dist)}    {self.make_bar(right_dist)}")
            print()
            print(f"              SAU ({back_pts} diem)")
            print(f"         {self.make_bar(back_dist)}")
            
            print()
            print("-" * 60)
            
            # Cảnh báo vật cản
            print(f"  Vat can gan nhat toan cuc: {global_min:.3f} m")
            print()
            
            if front_dist < 0.30:
                print("  >>> CANH BAO: VAT CAN RAT GAN PHIA TRUOC! <<<")
            elif front_dist < 0.50:
                print(f"  >> Phat hien vat can phia truoc: {front_dist:.2f}m - Can than!")
            elif front_dist < 1.00:
                print(f"  > Vat can phia truoc: {front_dist:.2f}m - Theo doi.")
            else:
                print("  Phia truoc: THONG THOANG")
            
            if left_dist < 0.30:
                print("  >> Vat can rat gan BEN TRAI!")
            if right_dist < 0.30:
                print("  >> Vat can rat gan BEN PHAI!")
                
            print("-" * 60)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        monitor = LidarMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nDa dung theo doi LiDAR.")
