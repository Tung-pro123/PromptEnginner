#!/usr/bin/env python3
"""
V3 Control — Topological Turn-Event Sequence Manager (Máy trạng thái sự kiện cua tuần tự)

Quy tắc bất biến theo cấu trúc sa bàn:
[1. THẲNG XUẤT PHÁT] ➔ [2. CUA TRÁI 1] ➔ [3. CUA PHẢI 1 (Chữ S)] ➔ [4. CUA TRÁI 2 (Thoát S)] ➔ [5. CUA PHẢI 2 (Về đích)] ➔ [LẶP LẠI VÒNG TIẾP THEO]

KHÔNG PHỤ THUỘC VÀO THỜI GIAN HAY CHIỀU DÀI:
Chỉ chuyển trạng thái khi xe thực sự chạm vào sự kiện đổi hướng cua của sa bàn!
"""

import time
import math
from enum import Enum
from dataclasses import dataclass


class TrackPhase(Enum):
    PHASE_1_STRAIGHT = "1.THANG_XUAT_PHAT"       # 1. Đoạn thẳng xuất phát (100% Ga)
    PHASE_2_TURN_LEFT = "2.CUA_TRAI_1"           # 2. Cua Trái đầu tiên (60% Ga)
    PHASE_3_TURN_RIGHT_S = "3.CUA_PHAI_CHU_S"    # 3. Cua Phải vào chữ S (35% Ga - An toàn)
    PHASE_4_TURN_LEFT_S = "4.CUA_TRAI_THOAT_S"   # 4. Cua Trái thoát chữ S (35% Ga - Không bứt tốc)
    PHASE_5_TURN_RIGHT_LOOP = "5.CUA_PHAI_VE_DICH" # 5. Cua Phải ôm về đích (60% Ga)


# Alias backward compatibility
TrackSector = TrackPhase


@dataclass
class PhaseControlProfile:
    """Tập quy tắc lái và ga cho từng pha cua."""
    phase: TrackPhase
    target_throttle: float       # Mức ga [0.0, 1.0]
    lookahead_m: float           # Khoảng cách ngắm Pure Pursuit
    feedforward_gain: float      # Hệ số bù góc lái
    feedforward_sign: float      # +1.0 (Phải), -1.0 (Trái), 0.0 (Thẳng)
    max_steer_limit: float = 1.00 # Giới hạn góc bẻ lái tối đa (100% công suất servo)
    status_desc: str = ""        # Mô tả


class TrackSectorManager:
    """Máy trạng thái sự kiện cua tuần tự (Event-Driven Turn Sequence)."""

    def __init__(self, config):
        self.cfg = config
        self.current_phase = TrackPhase.PHASE_1_STRAIGHT
        self.phase_start_time = time.time()
        self.lap_count = 0
        self.lap_start_time = time.time()
        self.last_lap_time = 0.0

        # Debounce chống nhiễu chuyển pha (Cần ít nhất 4 frame liên tiếp nhận đúng chiều cua)
        self._turn_debounce = 0
        self._debounce_required = 4

        # Cấu hình ga tối đa (Lấy trực tiếp từ config)
        self.max_speed = getattr(config, 'max_speed', 1.00)
        self.cruise_speed = getattr(config, 'cruise_speed', 0.68)
        self.corner_safe_speed = getattr(config, 'corner_safe_speed', 0.36)

    def update(self, curvature: float, max_upcoming_curvature: float = None, heading_error_deg: float = 0.0) -> PhaseControlProfile:
        """Chuyển pha dựa trên sự kiện đổi hướng độ cong thực tế của xe."""
        now = time.time()
        eff_k = max_upcoming_curvature if max_upcoming_curvature is not None else curvature

        # ------------------------------------------------------------------
        # MÁY CHUYỂN PHA TUẦN TỰ (PURE TURN-EVENT TRANSITIONS)
        # ------------------------------------------------------------------

        # PHA 1: THẲNG XUẤT PHÁT ──► PHA 2: CUA TRÁI 1
        if self.current_phase == TrackPhase.PHASE_1_STRAIGHT:
            # Nhận diện đường bắt đầu CUA TRÁI (curvature < -0.40)
            if curvature < -0.40 or eff_k < -0.40:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self._switch_phase(TrackPhase.PHASE_2_TURN_LEFT, now)
            else:
                self._turn_debounce = 0

        # PHA 2: CUA TRÁI 1 ──► PHA 3: CUA PHẢI VÀO CHỮ S
        elif self.current_phase == TrackPhase.PHASE_2_TURN_LEFT:
            # Nhận diện đảo dấu sang CUA PHẢI (curvature > +0.40)
            if curvature > 0.40 or eff_k > 0.40:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self._switch_phase(TrackPhase.PHASE_3_TURN_RIGHT_S, now)
            else:
                self._turn_debounce = 0

        # PHA 3: CUA PHẢI CHỮ S ──► PHA 4: CUA TRÁI THOÁT CHỮ S
        elif self.current_phase == TrackPhase.PHASE_3_TURN_RIGHT_S:
            # Nhận diện đảo dấu ngược lại sang CUA TRÁI (curvature < -0.40)
            if curvature < -0.40 or eff_k < -0.40:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self._switch_phase(TrackPhase.PHASE_4_TURN_LEFT_S, now)
            else:
                self._turn_debounce = 0

        # PHA 4: CUA TRÁI THOÁT CHỮ S ──► PHA 5: CUA PHẢI ÔM VỀ ĐÍCH
        elif self.current_phase == TrackPhase.PHASE_4_TURN_LEFT_S:
            # Nhận diện đảo dấu sang CUA PHẢI (curvature > +0.40)
            if curvature > 0.40 or eff_k > 0.40:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self._switch_phase(TrackPhase.PHASE_5_TURN_RIGHT_LOOP, now)
            else:
                self._turn_debounce = 0

        # PHA 5: CUA PHẢI VỀ ĐÍCH ──► PHA 1: THẲNG XUẤT PHÁT (HOÀN THÀNH 1 VÒNG!)
        elif self.current_phase == TrackPhase.PHASE_5_TURN_RIGHT_LOOP:
            # Nhận diện đường THẲNG TẮP trở lại (|curvature| < 0.25)
            if abs(curvature) < 0.25 and abs(heading_error_deg) < 12.0:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self.lap_count += 1
                    self.last_lap_time = now - self.lap_start_time
                    self.lap_start_time = now
                    print(f"🏁 [LAP #{self.lap_count}] Hoàn thành 1 vòng trong: {self.last_lap_time:.2f}s!")
                    self._switch_phase(TrackPhase.PHASE_1_STRAIGHT, now)
            else:
                self._turn_debounce = 0

        return self._get_profile()

    def _switch_phase(self, new_phase: TrackPhase, now: float):
        """Chuyển sang pha tiếp theo."""
        print(f"🚦 [TRACK EVENT] ➔ Chuyển sang: {new_phase.value} (Lap #{self.lap_count + 1})")
        self.current_phase = new_phase
        self.phase_start_time = now
        self._turn_debounce = 0

    def _get_profile(self) -> PhaseControlProfile:
        """Xuất quy tắc lái & ga cho từng pha chính xác."""
        p = self.current_phase

        # 1. ĐOẠN THẲNG XUẤT PHÁT: 100% Ga xé gió, nhìn xa 0.80m, không bù lái ảo
        if p == TrackPhase.PHASE_1_STRAIGHT:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.max_speed,
                lookahead_m=0.80,
                feedforward_gain=0.0,
                feedforward_sign=0.0,
                status_desc="🚀 1. THẲNG (100% GA)"
            )

        # 2. CUA TRÁI 1: 60% Ga ôm cua mượt, nhìn 0.30m, bù lái Trái 40%
        elif p == TrackPhase.PHASE_2_TURN_LEFT:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.cruise_speed,
                lookahead_m=0.30,
                feedforward_gain=0.40,
                feedforward_sign=-1.0,
                status_desc="🔄 2. CUA TRÁI 1 (60% GA)"
            )

        # 3. CUA PHẢI CHỮ S: 35% Ga an toàn tuyệt đối, nhìn 0.24m, bù lái Phải 45%
        elif p == TrackPhase.PHASE_3_TURN_RIGHT_S:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.corner_safe_speed,
                lookahead_m=0.24,
                feedforward_gain=0.45,
                feedforward_sign=1.0,
                status_desc="🛑 3. CUA PHẢI CHỮ S (35% GA)"
            )

        # 4. CUA TRÁI THOÁT CHỮ S: 35% Ga GHIM CỨNG (KHÔNG BỨT TỐC), nhìn 0.24m, bù lái Trái 45%
        elif p == TrackPhase.PHASE_4_TURN_LEFT_S:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.corner_safe_speed,
                lookahead_m=0.24,
                feedforward_gain=0.45,
                feedforward_sign=-1.0,
                status_desc="🛑 4. CUA TRÁI THOÁT CHỮ S (35% GA)"
            )

        # 5. CUA PHẢI ÔM VỀ ĐÍCH: 60% Ga ôm cua, nhìn 0.32m, bù lái Phải 40%
        elif p == TrackPhase.PHASE_5_TURN_RIGHT_LOOP:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.cruise_speed,
                lookahead_m=0.32,
                feedforward_gain=0.40,
                feedforward_sign=1.0,
                status_desc="🏁 5. CUA PHẢI VỀ ĐÍCH (60% GA)"
            )

        return PhaseControlProfile(
            phase=TrackPhase.PHASE_1_STRAIGHT,
            target_throttle=self.corner_safe_speed,
            lookahead_m=0.30,
            feedforward_gain=0.30,
            feedforward_sign=0.0,
            status_desc="TRACKING"
        )
