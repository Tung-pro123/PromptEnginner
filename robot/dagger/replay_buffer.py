#!/usr/bin/env python3
"""
DAgger — Replay Buffer
======================
Thread-safe circular buffer lưu trữ cặp (state, action) cho online learning.

Chiến lược sampling (chống Catastrophic Forgetting):
  - 50% từ NEW data  : dữ liệu Joy intervention mới nhất
  - 50% từ ANCHOR    : dữ liệu gốc load từ CSV logs/ (bộ nhớ nền)

Format:
  state  : np.ndarray (STATE_DIM,)  = [e_y, theta_e, line_visible, d1..d5, obstacle_detected]
  action : np.ndarray (2,)          = [steer, throttle]

CSV columns: timestamp, e_y, theta_e, line_visible, d_left, d_front_left,
             d_front, d_front_right, d_right, obstacle_detected, steer, throttle
"""

import os
import csv
import time
import threading
import numpy as np
from collections import deque

from robot.dagger.state_extractor import STATE_DIM

# =====================================================================
ACTION_DIM   = 2
BUFFER_MAXLEN = 5000    # Tối đa bao nhiêu mẫu mới trong buffer (circular)
ANCHOR_RATIO  = 0.5     # Tỉ lệ anchor trong mỗi batch


class ReplayBuffer:
    """
    Thread-safe Replay Buffer với anchor mixing.

    Usage:
        buf = ReplayBuffer(anchor_csv_path="logs/anchor_data.csv")
        buf.push(state, action)
        states, actions = buf.sample(batch_size=64)
    """

    def __init__(self, anchor_csv_path=None, maxlen=BUFFER_MAXLEN):
        self._lock  = threading.Lock()
        self._buf   = deque(maxlen=maxlen)   # dữ liệu mới (Joy interventions)
        self._anchor_states  = None          # np.ndarray (N, STATE_DIM)
        self._anchor_actions = None          # np.ndarray (N, ACTION_DIM)

        if anchor_csv_path and os.path.exists(anchor_csv_path):
            self._load_anchor(anchor_csv_path)
            print(f"[ReplayBuffer] Đã load {len(self._anchor_states)} anchor samples từ {anchor_csv_path}")
        else:
            print("[ReplayBuffer] Không có anchor data — sẽ học thuần từ Joy.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def push(self, state: np.ndarray, action: np.ndarray):
        """Đẩy 1 cặp (state, action) vào buffer."""
        assert state.shape == (STATE_DIM,),  f"state shape sai: {state.shape}"
        assert action.shape == (ACTION_DIM,), f"action shape sai: {action.shape}"
        with self._lock:
            self._buf.append((state.copy(), action.copy()))

    def sample(self, batch_size: int):
        """
        Sample một mini-batch với anchor mixing.

        Returns:
            states  : np.ndarray (batch_size, STATE_DIM)
            actions : np.ndarray (batch_size, ACTION_DIM)
            Trả về None, None nếu buffer quá ít dữ liệu.
        """
        with self._lock:
            n_new = len(self._buf)

        if n_new == 0 and self._anchor_states is None:
            return None, None

        # Số lượng anchor và new trong batch
        if self._anchor_states is not None:
            n_anchor = max(1, int(batch_size * ANCHOR_RATIO))
            n_new_sample = batch_size - n_anchor
        else:
            n_anchor = 0
            n_new_sample = batch_size

        states_list  = []
        actions_list = []

        # --- Lấy anchor samples ---
        if n_anchor > 0 and self._anchor_states is not None:
            idx = np.random.choice(len(self._anchor_states), size=n_anchor, replace=True)
            states_list.append(self._anchor_states[idx])
            actions_list.append(self._anchor_actions[idx])

        # --- Lấy new samples ---
        with self._lock:
            buf_list = list(self._buf)
        n_available = len(buf_list)
        if n_available > 0 and n_new_sample > 0:
            idx = np.random.choice(n_available,
                                   size=min(n_new_sample, n_available),
                                   replace=(n_new_sample > n_available))
            s = np.array([buf_list[i][0] for i in idx], dtype=np.float32)
            a = np.array([buf_list[i][1] for i in idx], dtype=np.float32)
            states_list.append(s)
            actions_list.append(a)

        if not states_list:
            return None, None

        states  = np.concatenate(states_list,  axis=0)
        actions = np.concatenate(actions_list, axis=0)
        return states, actions

    def __len__(self):
        with self._lock:
            return len(self._buf)

    def save_csv(self, path: str):
        """Lưu toàn bộ buffer mới ra CSV để dùng làm anchor sau này."""
        with self._lock:
            buf_list = list(self._buf)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'timestamp',
                'e_y', 'theta_e', 'line_visible',
                'd_left', 'd_front_left', 'd_front', 'd_front_right', 'd_right',
                'obstacle_detected',
                'steer', 'throttle'
            ])
            ts = time.time()
            for state, action in buf_list:
                w.writerow([ts] + state.tolist() + action.tolist())

        print(f"[ReplayBuffer] Đã lưu {len(buf_list)} mẫu → {path}")

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _load_anchor(self, csv_path: str):
        """Load anchor data từ CSV file (format giống save_csv)."""
        states, actions = [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    state = np.array([
                        float(row['e_y']),
                        float(row['theta_e']),
                        float(row['line_visible']),
                        float(row['d_left']),
                        float(row['d_front_left']),
                        float(row['d_front']),
                        float(row['d_front_right']),
                        float(row['d_right']),
                        float(row['obstacle_detected']),
                    ], dtype=np.float32)
                    action = np.array([
                        float(row['steer']),
                        float(row['throttle']),
                    ], dtype=np.float32)
                    states.append(state)
                    actions.append(action)
                except (KeyError, ValueError):
                    continue  # bỏ qua dòng lỗi

        if states:
            self._anchor_states  = np.array(states,  dtype=np.float32)
            self._anchor_actions = np.array(actions, dtype=np.float32)
        else:
            self._anchor_states  = None
            self._anchor_actions = None
