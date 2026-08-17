#!/usr/bin/env python3
import sys
import os

# Đảm bảo import được các module trong src/speed_track
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.speed_track.speed_racing_v3_1 import main

if __name__ == "__main__":
    print("=== Khởi chạy Phiên bản Chính (V3.1) ===")
    main()
