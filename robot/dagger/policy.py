#!/usr/bin/env python3
"""
DAgger — Policy Network (V2 - Dual-Head MLP with LayerNorm)
===========================================================
Mạng nơ-ron chính sách nhẹ nhưng giàu năng lực biểu diễn, tối ưu hóa cho
cả bám làn tốc độ cao và né vật cản mượt mà.

Kiến trúc:
  Input (STATE_DIM=15)
    │
    ▼
  Linear(15 ➔ 128) ➔ LayerNorm(128) ➔ ReLU ➔ Dropout(0.1)
    │
    ▼
  Linear(128 ➔ 64) ➔ ReLU
    │
    ├── Steer Head:    Linear(64 ➔ 32) ➔ ReLU ➔ Linear(32 ➔ 1) ➔ Tanh    ➔ Steer ∈ [-1.0, 1.0]
    └── Throttle Head: Linear(64 ➔ 32) ➔ ReLU ➔ Linear(32 ➔ 1) ➔ Sigmoid ➔ Throttle ∈ [0.0, 1.0]

Đặc tính:
  - Inference siêu tốc: ~0.5ms trên CPU Jetson Nano.
  - Tách bạch 2 heads: Steer Head chuyên trị góc bẻ lái né vật cản,
    Throttle Head học cách tự động giảm ga khi đến gần vật cản và bứt tốc ở đường thẳng.
  - LayerNorm giúp gradient ổn định khi nạp dữ liệu online liên tục.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robot.dagger.state_extractor import STATE_DIM

ACTION_DIM    = 2
HIDDEN_DIM_1  = 128
HIDDEN_DIM_2  = 64
HEAD_DIM      = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-5


# =====================================================================
# NETWORK DEFINITION
# =====================================================================

class _PolicyNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        # Backbone trích xuất đặc trưng chung
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM_1),
            nn.LayerNorm(HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
        )

        # Nhánh chuyên biệt học góc lái (Steer)
        self.steer_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, HEAD_DIM),
            nn.ReLU(),
            nn.Linear(HEAD_DIM, 1),
            nn.Tanh()  # [-1.0, 1.0]
        )

        # Nhánh chuyên biệt học vận tốc/ga (Throttle)
        self.throttle_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, HEAD_DIM),
            nn.ReLU(),
            nn.Linear(HEAD_DIM, 1),
            nn.Sigmoid()  # [0.0, 1.0]
        )

    def forward(self, x):
        """
        Args:
            x: (batch, STATE_DIM) tensor
        Returns:
            out: (batch, 2) — [steer, throttle]
        """
        features = self.backbone(x)
        steer = self.steer_head(features)
        throttle = self.throttle_head(features)
        return torch.cat([steer, throttle], dim=1)


# =====================================================================
# POLICY WRAPPER
# =====================================================================

class DAggerPolicy:
    """
    Wrapper quản lý toàn bộ vòng đời của Policy:
      - predict(state) → (steer, throttle)
      - update(states_batch, actions_batch) → loss
      - save(path) / load(path)
    """

    def __init__(self, model_path=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.net = _PolicyNet().to(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)

        self.update_count = 0
        self.last_loss = float('nan')

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            print(f"[Policy] Đã load model thành công từ {model_path}")
        else:
            print(f"[Policy] Khởi tạo model mới (STATE_DIM={STATE_DIM}).")

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def predict(self, state: np.ndarray):
        """
        Dự đoán hành động nhanh (không tính gradient, thread-safe).

        Args:
            state: np.ndarray shape (STATE_DIM,)
        Returns:
            steer    (float) ∈ [-1, 1]
            throttle (float) ∈ [0, 1]
        """
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self.net(x)
        steer = float(out[0, 0].item())
        throttle = float(out[0, 1].item())
        return steer, throttle

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def update(self, states: np.ndarray, actions: np.ndarray):
        """
        Một bước cập nhật gradient từ ReplayBuffer batch.

        Args:
            states  : (batch, STATE_DIM)
            actions : (batch, 2) — [steer, throttle]
        Returns:
            loss (float)
        """
        self.net.train()
        x = torch.tensor(states, dtype=torch.float32).to(self.device)
        y = torch.tensor(actions, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        preds = self.net(x)

        # Loss kết hợp: Steer loss + Throttle loss
        loss = self.loss_fn(preds, y)
        loss.backward()

        # Gradient clipping chống bùng nổ khi can thiệp gắt
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.update_count += 1
        self.last_loss = float(loss.item())
        return self.last_loss

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'state_dict': self.net.state_dict(),
            'update_count': self.update_count,
            'state_dim': STATE_DIM,
            'action_dim': ACTION_DIM,
        }, path)
        print(f"[Policy] Saved → {path} (total updates: {self.update_count})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            self.net.load_state_dict(ckpt['state_dict'])
            self.update_count = ckpt.get('update_count', 0)
        else:
            self.net.load_state_dict(ckpt)
            self.update_count = 0
        print(f"[Policy] Loaded ← {path} (updates: {self.update_count})")
