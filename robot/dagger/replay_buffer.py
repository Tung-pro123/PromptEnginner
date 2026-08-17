#!/usr/bin/env python3
"""
DAgger — Stratified Replay Buffer (V2 - Event-Balanced Sampling)
===============================================================
Thread-safe buffer lưu trữ cặp (state, action) cho online & offline learning.

Chiến lược lấy mẫu phân tầng (Stratified Sampling):
  1. ANCHOR DATA (40% batch) : Dữ liệu chuẩn nạp sẵn từ logs/anchor_data.csv
  2. DODGE DATA  (35% batch) : Các mẫu né vật cản / can thiệp góc lái gắt (|steer| > 0.35 hoặc có vật cản)
  3. NORMAL DATA (25% batch) : Các mẫu bám làn đi thẳng bình thường

Lợi ích:
  - Khắc phục triệt để hiện tượng mất cân bằng dữ liệu (Imbalance), đảm bảo AI
    không bị "lười né" vật cản dù số lần gặp vật cản ít hơn nhiều so với đi thẳng.
  - Chống quên thảm họa (Catastrophic Forgetting) nhờ bộ nhớ nền Anchor.

Format:
  state  : np.ndarray (STATE_DIM=15,)
  action : np.ndarray (ACTION_DIM=2,) = [steer, throttle]
"""

import os
import csv
import time
import threading
import numpy as np
from collections import deque

from robot.dagger.state_extractor import STATE_DIM

ACTION_DIM    = 2
BUFFER_MAXLEN = 5000     # Tối đa bao nhiêu mẫu trong mỗi deque
ANCHOR_RATIO  = 0.40     # 40% batch từ anchor
DODGE_RATIO   = 0.35     # 35% batch từ các mẫu né vật cản
NORMAL_RATIO  = 0.25     # 25% batch từ các mẫu bám làn thường

CSV_HEADERS_15D = [
    'timestamp',
    'e_y', 'e_y_dot', 'theta_e', 'curvature', 'line_visible',
    'd_left', 'd_front_left', 'd_front', 'd_front_right', 'd_right',
    'side_diff', 'min_front_dist', 'obstacle_detected',
    'prev_steer', 'prev_throttle',
    'steer', 'throttle'
]


class ReplayBuffer:
    """
    Thread-safe Replay Buffer phân tầng ưu tiên mẫu né vật cản.
    """

    def __init__(self, anchor_csv_path=None, maxlen=BUFFER_MAXLEN):
        self._lock = threading.Lock()
        self._normal_buf = deque(maxlen=maxlen)  # Dữ liệu đi thẳng / bám làn thường
        self._dodge_buf  = deque(maxlen=maxlen)  # Dữ liệu né vật cản / can thiệp góc lớn

        self._anchor_states  = None              # np.ndarray (N, STATE_DIM)
        self._anchor_actions = None              # np.ndarray (N, ACTION_DIM)

        if anchor_csv_path and os.path.exists(anchor_csv_path):
            self._load_anchor(anchor_csv_path)
            if self._anchor_states is not None:
                print(f"[ReplayBuffer] Đã load {len(self._anchor_states)} anchor samples từ {anchor_csv_path}")
        else:
            print("[ReplayBuffer] Chưa có anchor data — sẽ tích luỹ thuần từ Joy.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def push(self, state: np.ndarray, action: np.ndarray):
        """
        Đẩy 1 cặp (state, action) vào buffer phù hợp (phân loại tự động).
        """
        assert state.shape == (STATE_DIM,), f"state shape sai: {state.shape} (cần {STATE_DIM})"
        assert action.shape == (ACTION_DIM,), f"action shape sai: {action.shape}"

        steer = abs(float(action[0]))
        # state[12] là obstacle_detected, state[11] là min_front_dist
        has_obstacle = (state[12] > 0.5) or (state[11] < 0.55) or (steer > 0.35)

        with self._lock:
            if has_obstacle:
                self._dodge_buf.append((state.copy(), action.copy()))
            else:
                self._normal_buf.append((state.copy(), action.copy()))

    def sample(self, batch_size: int = 64):
        """
        Sample một mini-batch phân tầng cân bằng.

        Returns:
            states  : np.ndarray (batch_size, STATE_DIM)
            actions : np.ndarray (batch_size, ACTION_DIM)
            hoặc (None, None) nếu chưa đủ dữ liệu
        """
        with self._lock:
            n_normal = len(self._normal_buf)
            n_dodge  = len(self._dodge_buf)
            normal_list = list(self._normal_buf)
            dodge_list  = list(self._dodge_buf)

        n_anchor = len(self._anchor_states) if self._anchor_states is not None else 0
        total_available = n_normal + n_dodge + n_anchor

        if total_available < 10:
            return None, None

        # Tính toán phân bổ số lượng từng loại trong batch
        num_anchor = int(batch_size * ANCHOR_RATIO) if n_anchor > 0 else 0
        remaining = batch_size - num_anchor

        if n_dodge > 0 and n_normal > 0:
            num_dodge = int(remaining * (DODGE_RATIO / (DODGE_RATIO + NORMAL_RATIO)))
            num_normal = remaining - num_dodge
        elif n_dodge > 0:
            num_dodge = remaining
            num_normal = 0
        else:
            num_dodge = 0
            num_normal = remaining

        states_list = []
        actions_list = []

        # 1. Lấy từ Anchor
        if num_anchor > 0 and self._anchor_states is not None:
            idx = np.random.choice(n_anchor, size=num_anchor, replace=(num_anchor > n_anchor))
            states_list.append(self._anchor_states[idx])
            actions_list.append(self._anchor_actions[idx])

        # 2. Lấy từ Dodge Buffer (Né vật cản)
        if num_dodge > 0 and n_dodge > 0:
            idx = np.random.choice(n_dodge, size=num_dodge, replace=(num_dodge > n_dodge))
            s = np.array([dodge_list[i][0] for i in idx], dtype=np.float32)
            a = np.array([dodge_list[i][1] for i in idx], dtype=np.float32)
            states_list.append(s)
            actions_list.append(a)

        # 3. Lấy từ Normal Buffer (Bám làn thường)
        if num_normal > 0 and n_normal > 0:
            idx = np.random.choice(n_normal, size=num_normal, replace=(num_normal > n_normal))
            s = np.array([normal_list[i][0] for i in idx], dtype=np.float32)
            a = np.array([normal_list[i][1] for i in idx], dtype=np.float32)
            states_list.append(s)
            actions_list.append(a)

        if not states_list:
            return None, None

        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        return states, actions

    def __len__(self):
        with self._lock:
            return len(self._normal_buf) + len(self._dodge_buf)

    @property
    def dodge_count(self):
        with self._lock:
            return len(self._dodge_buf)

    @property
    def normal_count(self):
        with self._lock:
            return len(self._normal_buf)

    def save_csv(self, path: str):
        """Lưu toàn bộ mẫu mới (cả normal và dodge) ra CSV."""
        with self._lock:
            all_samples = list(self._normal_buf) + list(self._dodge_buf)

        if not all_samples:
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADERS_15D)
            ts = time.time()
            for state, action in all_samples:
                w.writerow([ts] + state.tolist() + action.tolist())

        print(f"[ReplayBuffer] Đã lưu {len(all_samples)} mẫu ({len(self._dodge_buf)} dodge) → {path}")

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _load_anchor(self, csv_path: str):
        """Load anchor data từ CSV file 15D."""
        states, actions = [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Đọc định dạng 15D
                    if 'side_diff' in row:
                        state = np.array([
                            float(row['e_y']),
                            float(row['e_y_dot']),
                            float(row['theta_e']),
                            float(row['curvature']),
                            float(row['line_visible']),
                            float(row['d_left']),
                            float(row['d_front_left']),
                            float(row['d_front']),
                            float(row['d_front_right']),
                            float(row['d_right']),
                            float(row['side_diff']),
                            float(row['min_front_dist']),
                            float(row['obstacle_detected']),
                            float(row.get('prev_steer', 0.0)),
                            float(row.get('prev_throttle', 0.0)),
                        ], dtype=np.float32)
                    else:
                        # Tự động tương thích với định dạng cũ (9D ➔ 15D)
                        d_left = float(row['d_left'])
                        d_right = float(row['d_right'])
                        d_front = float(row['d_front'])
                        d_fl = float(row.get('d_front_left', 1.0))
                        d_fr = float(row.get('d_front_right', 1.0))
                        state = np.array([
                            float(row['e_y']),
                            0.0,  # e_y_dot
                            float(row['theta_e']),
                            0.0,  # curvature
                            float(row['line_visible']),
                            d_left, d_fl, d_front, d_fr, d_right,
                            float(np.clip(d_left - d_right, -1.0, 1.0)),
                            float(min(d_fl, d_front, d_fr)),
                            float(row['obstacle_detected']),
                            0.0, 0.0
                        ], dtype=np.float32)

                    action = np.array([
                        float(row['steer']),
                        float(row['throttle']),
                    ], dtype=np.float32)
                    states.append(state)
                    actions.append(action)
                except Exception:
                    continue

        if states:
            self._anchor_states = np.array(states, dtype=np.float32)
            self._anchor_actions = np.array(actions, dtype=np.float32)
        else:
            self._anchor_states = None
            self._anchor_actions = None
