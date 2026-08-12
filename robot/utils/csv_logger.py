#!/usr/bin/env python3
"""
Module ghi Log CSV chuẩn BTC cho cuộc thi JetRacer 2026.

Ghi log theo đúng format quy định trong Đề bài chi tiết (Section 7):
- timestamp, fps, detected_object/sign, confidence, decision,
  latency_ms, control_output, event
"""

import os
import csv
import time


class CSVLogger:
    """Ghi log file .csv theo chuẩn BTC quy định.
    
    Format cột (theo đề bài Section 7):
        timestamp       - Thời điểm phát sinh sự kiện
        fps             - FPS hiện tại hoặc trung bình pipeline
        detected_object - Đối tượng nhận diện (lane, obstacle, checkpoint, ...)
        confidence      - Độ tin cậy (nếu có, 0-1)
        decision        - Quyết định điều khiển (keep_lane, avoid_left, avoid_right, ...)
        latency_ms      - Thời gian xử lý (ms)
        control_output  - Lệnh điều khiển (left_speed, right_speed)
        event           - Sự kiện đặc biệt (checkpoint_passed, lane_lost, ...)
    """

    FIELDNAMES = [
        'timestamp', 'fps', 'detected_object', 'confidence',
        'decision', 'latency_ms', 'control_output', 'event'
    ]

    def __init__(self, log_dir='logs', prefix='speed_track'):
        """
        Args:
            log_dir: Thư mục chứa file log
            prefix: Tiền tố tên file log
        """
        # Tạo thư mục nếu chưa có
        os.makedirs(log_dir, exist_ok=True)

        # Tạo tên file theo timestamp
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'{prefix}_{timestamp_str}.csv')

        # Mở file và ghi header
        self._file = open(self.log_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

        # Theo dõi FPS
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self._total_frames = 0
        self._total_time_start = time.time()

    def log(self, detected_object='', confidence=0.0, decision='',
            latency_ms=0.0, control_output='', event=''):
        """Ghi một dòng log.
        
        Args:
            detected_object: Đối tượng phát hiện (vd: 'lane', 'obstacle', 'checkpoint_1')
            confidence: Độ tin cậy (0.0 - 1.0)
            decision: Quyết định (vd: 'keep_lane', 'avoid_left', 'slow_down')
            latency_ms: Thời gian xử lý frame (milliseconds)
            control_output: Lệnh điều khiển (vd: 'L=0.27,R=0.25')
            event: Sự kiện đặc biệt (vd: 'checkpoint_1_passed', 'obstacle_avoided')
        """
        # Cập nhật FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

        row = {
            'timestamp': f'{time.time():.3f}',
            'fps': f'{self._current_fps:.1f}',
            'detected_object': detected_object,
            'confidence': f'{confidence:.2f}' if confidence > 0 else '',
            'decision': decision,
            'latency_ms': f'{latency_ms:.1f}',
            'control_output': control_output,
            'event': event,
        }
        self._writer.writerow(row)

        # Flush mỗi 20 dòng để tránh mất dữ liệu nếu crash
        if self._frame_count % 20 == 0:
            self._file.flush()

    def log_event(self, event_name, details=''):
        """Ghi log sự kiện đặc biệt (checkpoint, lỗi, ...).
        
        Args:
            event_name: Tên sự kiện
            details: Chi tiết bổ sung
        """
        self.log(event=f'{event_name}: {details}' if details else event_name)

    def get_average_fps(self):
        """Tính FPS trung bình toàn bộ lượt chạy."""
        self._total_frames += 1
        total_elapsed = time.time() - self._total_time_start
        if total_elapsed > 0:
            return self._total_frames / total_elapsed
        return 0.0

    def get_current_fps(self):
        """Trả về FPS hiện tại."""
        return self._current_fps

    def close(self):
        """Đóng file log và ghi dòng tổng kết."""
        avg_fps = self.get_average_fps()
        self.log(event=f'SESSION_END: avg_fps={avg_fps:.1f}')
        self._file.flush()
        self._file.close()

    def __del__(self):
        """Đảm bảo file được đóng khi object bị hủy."""
        try:
            if hasattr(self, '_file') and not self._file.closed:
                self.close()
        except Exception:
            pass
