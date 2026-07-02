#!/usr/bin/env python3
"""
Kiểm tra việc sắp xếp sys.path để import được cả rospy (Python 2) 
và các thư viện Python 3 (như jetracer, cv2, numpy) không bị xung đột.
"""
import sys

print("1. sys.path ban đầu:")
for p in sys.path:
    print(f"  - {p}")

# Tách riêng các đường dẫn python2.7 (ROS) và đưa chúng xuống cuối cùng
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]

# Sắp xếp lại: Ưu tiên Python 3 trước, Python 2 (ROS) xếp sau cùng
sys.path = py3_paths + py2_paths

print("\n2. sys.path sau khi sắp xếp lại (Ưu tiên Python 3 lên trước):")
for p in sys.path:
    print(f"  - {p}")

print("\n3. Thử import các thư viện:")

try:
    import jetracer
    print("  [OK] Import jetracer thành công!")
    from jetracer.nvidia_racecar import NvidiaRacecar
    print("  [OK] Khởi tạo class NvidiaRacecar thành công!")
except Exception as e:
    print(f"  [ERROR] Lỗi import jetracer: {e}")

try:
    import rospy
    print("  [OK] Import rospy thành công!")
except Exception as e:
    print(f"  [ERROR] Lỗi import rospy: {e}")

try:
    from sensor_msgs.msg import LaserScan
    print("  [OK] Import sensor_msgs LaserScan thành công!")
except Exception as e:
    print(f"  [ERROR] Lỗi import sensor_msgs: {e}")
