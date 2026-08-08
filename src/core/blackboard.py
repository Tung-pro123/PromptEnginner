# -*- coding: utf-8 -*-
"""
Blackboard Pattern: Bộ nhớ dùng chung giữa các mô-đun (Perception, FSM, Control)
"""

import threading

class Blackboard:
    """
    Lớp lưu trữ dữ liệu tập trung thread-safe.
    """
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def has(self, key):
        with self._lock:
            return key in self._data

    def clear(self):
        with self._lock:
            self._data.clear()
