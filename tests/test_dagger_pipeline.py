#!/usr/bin/env python3
"""
Test Suite — DAgger Imitation Learning Pipeline
================================================
Kiểm thử toàn diện các thành phần của hệ thống DAgger 15 chiều:
1. State Extractor (15D shape, bounds, side_diff, min_front)
2. Policy Network (Dual-head MLP, inference, training step, save/load)
3. Stratified Replay Buffer (Dodge vs Normal separation, balanced sampling, CSV persistence)
4. Background Trainer (Thread safety, gradient descent updates)
5. Safety Layer (E-STOP & Linear Throttle Scaling)
"""

import os
import sys
import time
import shutil
import tempfile
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from robot.dagger.state_extractor import extract_state, extract_lidar_zones, STATE_DIM
from robot.dagger.policy import DAggerPolicy, _PolicyNet, ACTION_DIM
from robot.dagger.replay_buffer import ReplayBuffer
from robot.dagger.trainer import BackgroundTrainer
from robot.dagger.safety import SafetyLayer


class MockLaserScan:
    def __init__(self, ranges=None):
        self.angle_min = -np.pi
        self.angle_max = np.pi
        self.angle_increment = 2 * np.pi / 360
        self.range_min = 0.15
        self.range_max = 12.0
        if ranges is None:
            self.ranges = [1.5] * 360
        else:
            self.ranges = ranges


class MockLaneState:
    def __init__(self, lateral_error_m=0.05, heading_error=0.1, curvature=0.5):
        from robot.estimation.lane_state import TrackingState
        self.lateral_error_m = lateral_error_m
        self.heading_error = heading_error
        self.curvature = curvature
        self.tracking_state = TrackingState.TRACKING


def test_state_extractor():
    print("\n--- [TEST 1] State Extractor (15D) ---")
    lane = MockLaneState(lateral_error_m=0.12, heading_error=-0.2, curvature=0.8)
    scan = MockLaserScan()

    # Thử nghiệm có vật cản bên phải (zone 3/4 khoảng cách gần)
    scan.ranges[120] = 0.35

    s_t, info = extract_state(
        lane, scan,
        prev_steer=0.1, prev_throttle=0.25,
        prev_e_y=0.10, dt=0.033
    )

    assert s_t.shape == (STATE_DIM,), f"Shape không đúng: {s_t.shape} (cần {STATE_DIM})"
    assert STATE_DIM == 15, f"STATE_DIM phải là 15, hiện tại là {STATE_DIM}"
    assert not np.isnan(s_t).any(), "State chứa giá trị NaN!"
    assert -1.0 <= s_t[0] <= 1.0, f"e_y ngoài biên: {s_t[0]}"
    assert -1.0 <= s_t[2] <= 1.0, f"theta_e ngoài biên: {s_t[2]}"
    assert -1.0 <= s_t[3] <= 1.0, f"curvature ngoài biên: {s_t[3]}"
    assert s_t[4] == 1.0, f"line_visible phải là 1.0, nhận được: {s_t[4]}"
    assert -1.0 <= s_t[10] <= 1.0, f"side_diff ngoài biên: {s_t[10]}"

    print(f"[OK] State Extractor OK! s_t shape: {s_t.shape}")
    print(f"     e_y={s_t[0]:.3f}, e_y_dot={s_t[1]:.3f}, theta_e={s_t[2]:.3f}, curvature={s_t[3]:.3f}")
    print(f"     side_diff={s_t[10]:.3f}, min_front={s_t[11]:.3f}, obstacle={s_t[12]}")


def test_policy_network():
    print("\n--- [TEST 2] Policy Network (Dual-Head MLP) ---")
    tmp_dir = tempfile.mkdtemp()
    model_path = os.path.join(tmp_dir, "test_policy.pt")

    try:
        policy = DAggerPolicy(device='cpu')
        dummy_state = np.random.uniform(-1, 1, size=(STATE_DIM,)).astype(np.float32)

        # 1. Test Inference
        steer, throttle = policy.predict(dummy_state)
        assert -1.0 <= steer <= 1.0, f"Steer out of range: {steer}"
        assert 0.0 <= throttle <= 1.0, f"Throttle out of range: {throttle}"
        print(f"     Inference output: Steer={steer:+.4f}, Throttle={throttle:.4f}")

        # 2. Test Gradient Update
        batch_states = np.random.uniform(-1, 1, size=(32, STATE_DIM)).astype(np.float32)
        batch_actions = np.random.uniform(-1, 1, size=(32, ACTION_DIM)).astype(np.float32)
        batch_actions[:, 1] = np.clip(batch_actions[:, 1], 0, 1)

        loss = policy.update(batch_states, batch_actions)
        assert loss > 0 and not np.isnan(loss), f"Invalid loss: {loss}"
        assert policy.update_count == 1, "update_count chưa tăng!"
        print(f"     Training step OK — loss: {loss:.5f}")

        # 3. Test Save/Load
        steer_after_update, throttle_after_update = policy.predict(dummy_state)
        policy.save(model_path)
        assert os.path.exists(model_path), "File model không tồn tại sau khi save!"

        new_policy = DAggerPolicy(model_path=model_path, device='cpu')
        assert new_policy.update_count == 1, f"Load update_count sai: {new_policy.update_count}"
        new_steer, new_throttle = new_policy.predict(dummy_state)
        assert np.isclose(steer_after_update, new_steer, atol=1e-5), "Predict sau khi load bị lệch!"
        assert np.isclose(throttle_after_update, new_throttle, atol=1e-5), "Predict throttle sau khi load bị lệch!"
        print("[OK] Policy Network Save/Load & Inference OK!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_stratified_replay_buffer():
    print("\n--- [TEST 3] Stratified Replay Buffer ---")
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "test_buffer.csv")

    try:
        buf = ReplayBuffer(maxlen=500)

        # Đẩy 50 mẫu bình thường
        for _ in range(50):
            s = np.zeros(STATE_DIM, dtype=np.float32)
            s[11] = 1.0  # min_front xa
            s[12] = 0.0  # không có vật cản
            a = np.array([0.05, 0.25], dtype=np.float32)
            buf.push(s, a)

        # Đẩy 20 mẫu né vật cản
        for _ in range(20):
            s = np.zeros(STATE_DIM, dtype=np.float32)
            s[11] = 0.3  # min_front gần
            s[12] = 1.0  # cờ vật cản
            a = np.array([0.65, 0.15], dtype=np.float32)
            buf.push(s, a)

        assert buf.normal_count == 50, f"Normal count sai: {buf.normal_count}"
        assert buf.dodge_count == 20, f"Dodge count sai: {buf.dodge_count}"
        assert len(buf) == 70, f"Total len sai: {len(buf)}"

        # Sample batch
        states, actions = buf.sample(batch_size=32)
        assert states.shape == (32, STATE_DIM), f"States batch shape sai: {states.shape}"
        assert actions.shape == (32, ACTION_DIM), f"Actions batch shape sai: {actions.shape}"

        # Kiểm tra mẫu né trong batch
        dodge_in_batch = (actions[:, 0] > 0.35).sum()
        print(f"     Batch 32 mẫu có {dodge_in_batch} mẫu né vật cản ({dodge_in_batch/32*100:.1f}%)")
        assert dodge_in_batch > 0, "Không có mẫu né vật cản nào được lấy mẫu!"

        # Test Save CSV & Load
        buf.save_csv(csv_path)
        assert os.path.exists(csv_path), "File CSV không tồn tại sau khi save!"

        buf_loaded = ReplayBuffer(anchor_csv_path=csv_path)
        assert buf_loaded._anchor_states is not None
        assert len(buf_loaded._anchor_states) == 70
        print("[OK] Stratified Replay Buffer OK!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_background_trainer():
    print("\n--- [TEST 4] Background Trainer (Threading) ---")
    policy = DAggerPolicy(device='cpu')
    buf = ReplayBuffer(maxlen=500)

    # Đẩy 80 mẫu để vượt ngưỡng MIN_SAMPLES = 50
    for _ in range(80):
        s = np.random.uniform(-1, 1, size=(STATE_DIM,)).astype(np.float32)
        a = np.random.uniform(-1, 1, size=(ACTION_DIM,)).astype(np.float32)
        a[1] = np.clip(a[1], 0, 1)
        buf.push(s, a)

    trainer = BackgroundTrainer(policy, buf)
    trainer.start()
    assert trainer.is_alive(), "Trainer thread không khởi động được!"

    time.sleep(0.5)  # Chờ 0.5s cho trainer cập nhật
    trainer.stop()

    assert policy.update_count > 0, f"Trainer chưa thực hiện cập nhật nào! count={policy.update_count}"
    print(f"[OK] Background Trainer OK! Đã hoàn thành {policy.update_count} bước update trong background thread.")


def test_safety_layer():
    print("\n--- [TEST 5] Safety Layer ---")
    safety = SafetyLayer()

    # 1. Khi an toàn (zone = 1.0 -> 1.5m)
    zones = np.ones(5, dtype=np.float32)
    steer, thr, estop = safety.check(0.5, 0.3, zones)
    assert not estop and thr == 0.3, "Safety can thiệp nhầm khi đường thông!"

    # 2. Khi có vật cản gần (front = 0.15m / 1.5m = 0.1) -> E-STOP
    zones[2] = 0.10  # 0.15m < d_critical (0.25m)
    steer, thr, estop = safety.check(0.5, 0.3, zones)
    assert estop and thr == 0.0, "Safety không kích hoạt E-STOP khi vật cản quá gần!"

    # 3. Khi có vật cản ở vùng cảnh báo (front = 0.40m / 1.5m = 0.267) -> Giảm ga
    zones[2] = 0.40 / 1.5
    steer, thr, estop = safety.check(0.5, 0.3, zones)
    assert not estop and 0.0 < thr < 0.3, f"Safety không hạ ga hợp lý: thr={thr}"
    print("[OK] Safety Layer OK!")


if __name__ == '__main__':
    print("=========================================================")
    print("[RUN] CHAY TOAN BO KIEM THU PIPELINE DAGGER (15-DIMENSIONAL)")
    print("=========================================================")
    test_state_extractor()
    test_policy_network()
    test_stratified_replay_buffer()
    test_background_trainer()
    test_safety_layer()
    print("\n[SUCCESS] TAT CA CAC KIEM THU DA VUOT QUA XUAT SAC! [SUCCESS]")
