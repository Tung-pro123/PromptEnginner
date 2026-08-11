#!/usr/bin/env python3
"""
Module Theo dõi Checkpoint cho Speed Track.

Đề bài yêu cầu:
- 3 Checkpoint đánh số trên sa bàn, phải vượt qua theo đúng thứ tự (1 → 2 → 3)
- Vạch xuất phát đồng thời là vạch kết thúc
- Checkpoint hợp lệ khi toàn bộ 4 bánh xe đã vượt qua vạch

Phương pháp nhận diện checkpoint:
- Dùng camera: phát hiện vạch ngang đặc trưng hoặc marker trên sa bàn
- Dùng LiDAR: phát hiện cấu trúc đặc biệt 2 bên checkpoint (nếu có)
- Kết hợp cooldown để tránh đếm trùng
"""

import time


class CheckpointTracker:
    """Theo dõi và xác nhận việc vượt qua Checkpoint trên Speed Track.
    
    Scoring (theo đề bài Section 3.6):
    - Điểm Checkpoint = số CP hợp lệ × 10 điểm, tối đa 30 điểm
    - CP phải được vượt theo đúng thứ tự: 1 → 2 → 3
    """

    TOTAL_CHECKPOINTS = 3

    def __init__(self, cooldown_seconds=3.0):
        """
        Args:
            cooldown_seconds: Thời gian chờ giữa 2 lần nhận checkpoint
                             (tránh đếm trùng khi xe đi qua vạch checkpoint chậm)
        """
        self.cooldown_seconds = cooldown_seconds
        self.reset()

    def reset(self):
        """Reset bộ đếm checkpoint (dùng khi bắt đầu lượt chạy mới)."""
        self.checkpoints_passed = 0
        self.last_checkpoint_time = 0.0
        self.checkpoint_times = []  # Thời điểm vượt qua mỗi CP
        self.finished = False

    def try_register_checkpoint(self, current_time=None):
        """Thử đăng ký một checkpoint mới.
        
        Gọi hàm này khi hệ thống phát hiện vạch checkpoint (từ camera hoặc LiDAR).
        Hàm sẽ tự kiểm tra cooldown và thứ tự.
        
        Args:
            current_time: Thời gian hiện tại (seconds). Nếu None, dùng time.time().
            
        Returns:
            dict: {
                'registered': bool (True nếu CP được đăng ký thành công),
                'checkpoint_number': int (số thứ tự CP vừa vượt, 1-indexed),
                'total_passed': int (tổng số CP đã vượt),
                'all_passed': bool (True nếu đã vượt hết 3 CP)
            }
        """
        if current_time is None:
            current_time = time.time()

        # Kiểm tra đã hoàn thành chưa
        if self.finished:
            return {
                'registered': False,
                'checkpoint_number': 0,
                'total_passed': self.checkpoints_passed,
                'all_passed': True,
            }

        # Kiểm tra cooldown
        if current_time - self.last_checkpoint_time < self.cooldown_seconds:
            return {
                'registered': False,
                'checkpoint_number': 0,
                'total_passed': self.checkpoints_passed,
                'all_passed': False,
            }

        # Đăng ký checkpoint mới
        self.checkpoints_passed += 1
        self.last_checkpoint_time = current_time
        self.checkpoint_times.append(current_time)

        cp_number = self.checkpoints_passed

        # Kiểm tra đã vượt hết 3 CP chưa
        if self.checkpoints_passed >= self.TOTAL_CHECKPOINTS:
            self.finished = True

        return {
            'registered': True,
            'checkpoint_number': cp_number,
            'total_passed': self.checkpoints_passed,
            'all_passed': self.finished,
        }

    def get_score(self):
        """Tính điểm Checkpoint hiện tại.
        
        Returns:
            int: Điểm checkpoint (0, 10, 20, hoặc 30)
        """
        return min(self.checkpoints_passed, self.TOTAL_CHECKPOINTS) * 10

    def get_status(self):
        """Trả về trạng thái tổng quan.
        
        Returns:
            dict: Trạng thái checkpoint tracker
        """
        return {
            'passed': self.checkpoints_passed,
            'total': self.TOTAL_CHECKPOINTS,
            'remaining': max(0, self.TOTAL_CHECKPOINTS - self.checkpoints_passed),
            'score': self.get_score(),
            'finished': self.finished,
            'times': self.checkpoint_times[:],
        }
