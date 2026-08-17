#!/usr/bin/env python3
"""
Export DAgger Policy → ONNX
============================
Sau khi đã fine-tune xong, export model PyTorch sang ONNX để:
  - Deploy với ONNXRuntime / TensorRT (tăng tốc độ suy luận)
  - Tích hợp vào các pipeline điều khiển tự hành nếu cần

Usage:
    python3 training/export_dagger.py

Output: models/dagger_policy.onnx
"""

import os
import sys
import torch
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from robot.dagger.policy          import DAggerPolicy, _PolicyNet
from robot.dagger.state_extractor import STATE_DIM

MODEL_PT   = os.path.join(_ROOT, 'models', 'dagger_policy.pt')
MODEL_ONNX = os.path.join(_ROOT, 'models', 'dagger_policy.onnx')


def export():
    print(f"[Export] Load model từ: {MODEL_PT}")
    if not os.path.exists(MODEL_PT):
        print(f"[Export] KHÔNG TÌM THẤY {MODEL_PT}")
        print("[Export] Hãy chạy main_dagger.py trước để tạo model.")
        return

    # Load
    policy = DAggerPolicy(model_path=MODEL_PT)
    net    = policy.net
    net.eval()

    # Dummy input
    dummy_input = torch.zeros(1, STATE_DIM, dtype=torch.float32)

    # Export
    torch.onnx.export(
        net,
        dummy_input,
        MODEL_ONNX,
        export_params=True,
        opset_version=11,
        input_names  = ['state'],
        output_names = ['action'],
        dynamic_axes = {
            'state' : {0: 'batch'},
            'action': {0: 'batch'},
        },
    )
    print(f"[Export] Đã xuất ONNX → {MODEL_ONNX}")

    # Verify với onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(MODEL_ONNX, providers=['CPUExecutionProvider'])
        dummy_np = np.zeros((1, STATE_DIM), dtype=np.float32)
        out = sess.run(None, {'state': dummy_np})
        print(f"[Export] Kiểm tra ONNX OK — output shape: {out[0].shape}, values: {out[0]}")
    except ImportError:
        print("[Export] (Bỏ qua kiểm tra — onnxruntime chưa cài)")


if __name__ == '__main__':
    export()
