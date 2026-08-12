#!/usr/bin/env python3
"""
DAgger — Background Trainer
============================
Thread riêng liên tục lấy batch từ ReplayBuffer và fine-tune Policy.

Thiết kế:
  - Chạy trong daemon thread → tự chết khi main process kết thúc
  - Chỉ train khi buffer có >= MIN_SAMPLES mẫu (tránh học quá sớm)
  - Nghỉ SLEEP_INTERVAL giây giữa mỗi bước nếu train quá nhanh (CPU relief)
  - Dùng threading.Event để main loop có thể signal dừng sạch

Stats được in mỗi PRINT_EVERY steps để debug không làm rối console.
"""

import time
import threading

# Cấu hình trainer
BATCH_SIZE     = 64      # số mẫu mỗi bước gradient
MIN_SAMPLES    = 50      # cần ít nhất N mẫu trong buffer mới train
SLEEP_INTERVAL = 0.05    # giây nghỉ giữa các bước (20 steps/s tối đa)
PRINT_EVERY    = 100     # in stats mỗi N bước


class BackgroundTrainer:
    """
    Daemon thread thực hiện fine-tune liên tục.

    Usage:
        trainer = BackgroundTrainer(policy, replay_buffer)
        trainer.start()
        # ... chạy main loop ...
        trainer.stop()
    """

    def __init__(self, policy, replay_buffer):
        """
        Args:
            policy       : DAggerPolicy instance
            replay_buffer: ReplayBuffer instance
        """
        self.policy  = policy
        self.buffer  = replay_buffer
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="DAggerTrainer")

    def start(self):
        """Bắt đầu background training thread."""
        self._stop_event.clear()
        self._thread.start()
        print("[Trainer] Background training thread started.")

    def stop(self, timeout=3.0):
        """Signal dừng và chờ thread kết thúc."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        print(f"[Trainer] Thread stopped. Total updates: {self.policy.update_count}")

    def is_alive(self):
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # MAIN TRAINING LOOP (chạy trong thread riêng)
    # ------------------------------------------------------------------

    def _run(self):
        steps = 0
        while not self._stop_event.is_set():
            # Kiểm tra đủ dữ liệu chưa
            n_new = len(self.buffer)
            if n_new < MIN_SAMPLES:
                # Chưa đủ → ngủ ngắn rồi kiểm tra lại
                time.sleep(0.2)
                continue

            # Sample batch
            states, actions = self.buffer.sample(BATCH_SIZE)
            if states is None:
                time.sleep(0.2)
                continue

            # Một bước gradient descent
            try:
                loss = self.policy.update(states, actions)
            except Exception as e:
                print(f"[Trainer] Lỗi train step: {e}")
                time.sleep(0.5)
                continue

            steps += 1

            # Print stats
            if steps % PRINT_EVERY == 0:
                print(
                    f"[Trainer] step={steps:6d} | "
                    f"loss={loss:.5f} | "
                    f"buffer={n_new} | "
                    f"total_updates={self.policy.update_count}"
                )

            # Nghỉ ngắn để không chiếm hết CPU của main loop
            time.sleep(SLEEP_INTERVAL)
