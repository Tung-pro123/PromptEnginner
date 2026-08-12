#!/usr/bin/env python3
"""
DAgger — Policy (MLP)
=====================
Mạng neural nhỏ để inference và fine-tune online trên Jetson Nano CPU.

Kiến trúc:
  Input(STATE_DIM=9) → FC(64,ReLU) → Dropout(0.1) → FC(32,ReLU) → FC(2)
  Output[0] → steer   (tanh  → [-1, 1])
  Output[1] → throttle (sigmoid → [0, 1])

Lưu/load format: PyTorch .pt (state_dict)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robot.dagger.state_extractor import STATE_DIM

ACTION_DIM   = 2
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
LEARNING_RATE = 5e-4


# =====================================================================
# NETWORK
# =====================================================================

class _PolicyNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
        )
        self.steer_head    = nn.Linear(HIDDEN_DIM_2, 1)   # raw → tanh
        self.throttle_head = nn.Linear(HIDDEN_DIM_2, 1)   # raw → sigmoid

    def forward(self, x):
        """
        Args:
            x: (batch, STATE_DIM) tensor

        Returns:
            out: (batch, 2) — [steer, throttle]
        """
        features  = self.net(x)
        steer     = torch.tanh(self.steer_head(features))      # [-1, 1]
        throttle  = torch.sigmoid(self.throttle_head(features)) # [0, 1]
        return torch.cat([steer, throttle], dim=1)


# =====================================================================
# POLICY WRAPPER
# =====================================================================

class DAggerPolicy:
    """
    Wrapper giúp main loop dễ dùng:
      - predict(state) → (steer, throttle)
      - update(states_batch, actions_batch) → loss
      - save(path) / load(path)
    """

    def __init__(self, model_path=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.net = _PolicyNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn   = nn.SmoothL1Loss()

        # Đếm số lần update — để main loop biết đã train bao nhiêu step
        self.update_count = 0
        self.last_loss    = float('nan')

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            print(f"[Policy] Đã load model từ {model_path}")
        else:
            print(f"[Policy] Khởi tạo model mới (chưa có pretrained weights).")

    # ------------------------------------------------------------------
    # INFERENCE (thread-safe: không dùng grad)
    # ------------------------------------------------------------------

    def predict(self, state: np.ndarray):
        """
        Inference nhanh — không tính gradient.

        Args:
            state: np.ndarray shape (STATE_DIM,)

        Returns:
            steer    (float) ∈ [-1, 1]
            throttle (float) ∈ [0, 1]
        """
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self.net(x)         # (1, 2)
        steer    = float(out[0, 0].item())
        throttle = float(out[0, 1].item())
        return steer, throttle

    # ------------------------------------------------------------------
    # TRAINING (gọi từ BackgroundTrainer thread)
    # ------------------------------------------------------------------

    def update(self, states: np.ndarray, actions: np.ndarray):
        """
        Một bước fine-tune supervised với batch đã sample từ ReplayBuffer.

        Args:
            states  : (batch, STATE_DIM)
            actions : (batch, 2) — [steer_label, throttle_label]

        Returns:
            loss (float)
        """
        self.net.train()
        x = torch.tensor(states,  dtype=torch.float32).to(self.device)
        y = torch.tensor(actions, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        preds = self.net(x)         # (batch, 2)
        loss  = self.loss_fn(preds, y)
        loss.backward()

        # Gradient clipping — tránh phát nổ khi dữ liệu ít
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.update_count += 1
        self.last_loss     = loss.item()
        return self.last_loss

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict'   : self.net.state_dict(),
            'update_count' : self.update_count,
        }, path)
        print(f"[Policy] Saved → {path} (updates: {self.update_count})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['state_dict'])
        self.update_count = ckpt.get('update_count', 0)
        print(f"[Policy] Loaded ← {path} (updates: {self.update_count})")
