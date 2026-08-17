#!/usr/bin/env python3
"""
DAgger — Offline Pre-Training & Bootstrap Pipeline
==================================================
Huấn luyện mô hình DAgger Policy từ các file log CSV đã thu thập.

Tính năng:
  - Tự động nạp tất cả session CSV trong logs/dagger/ hoặc file chỉ định.
  - Data Augmentation (Left-Right Mirroring): Lật đối xứng gương nhân đôi dữ liệu,
    giúp chính sách lái hoàn toàn cân bằng giữa rẽ trái và rẽ phải.
  - Tự động lưu checkpoint `models/dagger_policy.pt` và xuất sang `models/dagger_policy.onnx`.

Usage:
    python3 training/train_dagger_offline.py [--epochs 200] [--lr 0.0005] [--batch-size 64]
"""

import os
import sys
import glob
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from robot.dagger.state_extractor import STATE_DIM
from robot.dagger.policy import _PolicyNet, DAggerPolicy

MODEL_PT_PATH = os.path.join(_ROOT, 'models', 'dagger_policy.pt')
MODEL_ONNX_PATH = os.path.join(_ROOT, 'models', 'dagger_policy.onnx')
LOG_DIR = os.path.join(_ROOT, 'logs', 'dagger')


# =====================================================================
# DATASET & AUGMENTATION
# =====================================================================

class DAggerOfflineDataset(Dataset):
    def __init__(self, data_paths, augment_mirror=True):
        """
        Args:
            data_paths: list các đường dẫn file CSV
            augment_mirror: True để nhân đôi dữ liệu bằng phép lật đối xứng gương
        """
        self.states = []
        self.actions = []

        total_files = 0
        for p in data_paths:
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            total_files += 1

            for _, row in df.iterrows():
                try:
                    # Trích xuất state 15D
                    if 'side_diff' in row:
                        s = [
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
                        ]
                    else:
                        d_l = float(row['d_left'])
                        d_r = float(row['d_right'])
                        d_f = float(row['d_front'])
                        d_fl = float(row.get('d_front_left', 1.0))
                        d_fr = float(row.get('d_front_right', 1.0))
                        s = [
                            float(row['e_y']),
                            0.0,
                            float(row['theta_e']),
                            0.0,
                            float(row['line_visible']),
                            d_l, d_fl, d_f, d_fr, d_r,
                            float(np.clip(d_l - d_r, -1.0, 1.0)),
                            float(min(d_fl, d_f, d_fr)),
                            float(row['obstacle_detected']),
                            0.0, 0.0
                        ]

                    steer = float(row.get('cmd_steer', row.get('steer', 0.0)))
                    throttle = float(row.get('cmd_throttle', row.get('throttle', 0.0)))
                    a = [steer, throttle]

                    s_arr = np.array(s, dtype=np.float32)
                    a_arr = np.array(a, dtype=np.float32)

                    self.states.append(s_arr)
                    self.actions.append(a_arr)

                    # --- Phép lật đối xứng gương (Mirror Augmentation) ---
                    if augment_mirror:
                        s_mir = s_arr.copy()
                        # Đảo dấu sai số lệch tâm, đạo hàm, lệch góc, độ cong
                        s_mir[0] = -s_mir[0]  # e_y
                        s_mir[1] = -s_mir[1]  # e_y_dot
                        s_mir[2] = -s_mir[2]  # theta_e
                        s_mir[3] = -s_mir[3]  # curvature
                        # Đổi chỗ khoảng cách LiDAR trái <-> phải
                        s_mir[5], s_mir[9] = s_arr[9], s_arr[5]   # d_left <-> d_right
                        s_mir[6], s_mir[8] = s_arr[8], s_arr[6]   # d_fl <-> d_fr
                        s_mir[10] = -s_mir[10]                    # side_diff
                        s_mir[13] = -s_mir[13]                    # prev_steer

                        a_mir = a_arr.copy()
                        a_mir[0] = -a_mir[0]                      # steer

                        self.states.append(s_mir)
                        self.actions.append(a_mir)

                except Exception:
                    continue

        print(f"[Dataset] Đã đọc {total_files} files CSV. Tổng số mẫu sau augmentation: {len(self.states)}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx], dtype=torch.float32), torch.tensor(self.actions[idx], dtype=torch.float32)


# =====================================================================
# TRAINING PIPELINE
# =====================================================================

def train_offline(epochs=200, batch_size=64, lr=5e-4, csv_dir=LOG_DIR):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print(f"[ERROR] Không tìm thấy file CSV nào trong {csv_dir}")
        print("Hãy chạy xe với tay cầm trước hoặc cung cấp file anchor_data.csv!")
        return

    print(f"[Offline Train] Tìm thấy {len(csv_files)} logs trong {csv_dir}")
    dataset = DAggerOfflineDataset(csv_files, augment_mirror=True)
    if len(dataset) == 0:
        print("[ERROR] Dataset trống!")
        return

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Offline Train] Sử dụng thiết bị: {device}")

    policy = DAggerPolicy(model_path=MODEL_PT_PATH if os.path.exists(MODEL_PT_PATH) else None, device=device)
    optimizer = optim.AdamW(policy.net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN ({epochs} Epochs) ---")
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        policy.net.train()
        total_loss = 0.0
        steer_err = 0.0
        throttle_err = 0.0

        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            preds = policy.net(states)

            loss = loss_fn(preds, actions)
            loss.backward()

            nn.utils.clip_grad_norm_(policy.net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * states.size(0)
            steer_err += torch.abs(preds[:, 0] - actions[:, 0]).sum().item()
            throttle_err += torch.abs(preds[:, 1] - actions[:, 1]).sum().item()

        scheduler.step()

        epoch_loss = total_loss / len(dataset)
        avg_steer_err = steer_err / len(dataset)
        avg_thr_err = throttle_err / len(dataset)

        if epoch % 20 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch [{epoch:3d}/{epochs:3d}] | Loss: {epoch_loss:.5f} | Steer MAE: {avg_steer_err:.4f} | Throttle MAE: {avg_thr_err:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            policy.save(MODEL_PT_PATH)

    print(f"\n🎉 Huấn luyện hoàn tất! Best Loss: {best_loss:.5f}")
    print(f"Trọng số đã được lưu tại: {MODEL_PT_PATH}")

    # Xuất mô hình ONNX
    export_to_onnx(policy.net, device)


def export_to_onnx(net, device):
    net.eval()
    dummy_input = torch.zeros(1, STATE_DIM, dtype=torch.float32).to(device)
    os.makedirs(os.path.dirname(MODEL_ONNX_PATH), exist_ok=True)

    torch.onnx.export(
        net,
        dummy_input,
        MODEL_ONNX_PATH,
        export_params=True,
        opset_version=11,
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={
            'state': {0: 'batch'},
            'action': {0: 'batch'},
        }
    )
    print(f"[Export ONNX] Đã xuất thành công sang: {MODEL_ONNX_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DAgger Offline Pre-training")
    parser.add_argument('--epochs', type=int, default=150, help="Số epochs huấn luyện")
    parser.add_argument('--batch-size', type=int, default=64, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--data-dir', type=str, default=LOG_DIR, help="Thư mục chứa logs CSV")

    args = parser.parse_args()
    train_offline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, csv_dir=args.data_dir)
