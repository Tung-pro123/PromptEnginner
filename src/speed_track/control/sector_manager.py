#!/usr/bin/env python3
"""
V3 Control — Topological Turn-Event Sequence Manager (Chu trình sa bàn B-Shape chuẩn xác)

CẤU TRÚC HÌNH HỌC VÒNG ĐUA NGƯỢC CHIỀU KIM ĐỒNG HỒ (CCW):
[1. THẲNG 05 ➔ 04] ──► [2. CUA TRÁI CUNG TRÊN 01] ──► [3. CUA PHẢI VÀO S 02] ──► [4. CUA TRÁI THOÁT S & ÔM CUNG DƯỚI 01] ──► [VỀ ĐÍCH 05 & LẶP LẠI]

ĐẶC BIỆT BẢO VỆ ĐOẠN 02 ➔ 01 CUNG DƯỚI:
Khóa chặt hướng cua TRÁI liên tục từ lúc thoát chữ S cho đến khi về đích,
TUYỆT ĐỐI không bị nhiễu bởi vạch tường bên phải và KHÔNG BAO GIỜ bẻ lái nhầm sang Phải!
"""

import time
import math
from enum import Enum
from dataclasses import dataclass


class TrackPhase(Enum):
    PHASE_1_STRAIGHT = "1.THANG_05->04"                       # 1. Đoạn thẳng xuất phát bứt tốc
    PHASE_2_TURN_LEFT_UPPER = "2.CUA_TRAI_CUNG_TREN_01"       # 2. Cua Trái cung tròn trên
    PHASE_3_TURN_RIGHT_S_ENTRY = "3.CUA_PHAI_VAO_CHU_S_02"    # 3. Cua Phải vào eo chữ S
    PHASE_4_TURN_LEFT_S_EXIT_LOWER = "4.CUA_TRAI_THOAT_S_VA_CUNG_DUOI_01" # 4. Cua Trái thoát S và ôm trọn cung dưới về đích


# Alias backward compatibility
TrackSector = TrackPhase


@dataclass
class PhaseControlProfile:
    """Tập quy tắc lái và ga cho từng pha cua chuẩn xác."""
    phase: TrackPhase
    target_throttle: float       # Mức ga [0.0, 1.0]
    lookahead_m: float           # Khoảng cách ngắm Pure Pursuit
    feedforward_gain: float      # Hệ số bù góc lái
    feedforward_sign: float      # +1.0 (Phải), -1.0 (Trái), 0.0 (Thẳng)
    max_steer_limit: float = 1.00 # Giới hạn góc bẻ lái tối đa (100% công suất servo)
    status_desc: str = ""        # Mô tả


class TrackSectorManager:
    """Máy trạng thái sự kiện cua tuần tự chống bắt nhầm vạch biên."""

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

        # Cấu hình ga
        self.max_speed = getattr(config, 'max_speed', 1.00)
        self.cruise_speed = getattr(config, 'cruise_speed', 0.68)
        self.corner_safe_speed = getattr(config, 'corner_safe_speed', 0.36)

    def update(self, curvature: float, max_upcoming_curvature: float = None, heading_error_deg: float = 0.0) -> PhaseControlProfile:
        """Chuyển pha dựa trên sự kiện đổi hướng độ cong thực tế của xe."""
        now = time.time()
        time_in_phase = now - self.phase_start_time
        eff_k = max_upcoming_curvature if max_upcoming_curvature is not None else curvature

        # ------------------------------------------------------------------
        # MÁY CHUYỂN PHA TÔ-PÔ BẢO VỆ CHỐNG BẮT NHẦM VẠCH
        # ------------------------------------------------------------------

        # PHA 1: THẲNG XUẤT PHÁT [05 -> 04] ──► PHA 2: CUA TRÁI CUNG TRÊN [01]
        if self.current_phase == TrackPhase.PHASE_1_STRAIGHT:
            # Nhận diện đường bắt đầu CUA TRÁI (curvature < -0.38)
            if curvature < -0.38 or eff_k < -0.38:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self._switch_phase(TrackPhase.PHASE_2_TURN_LEFT_UPPER, now)
            else:
                self._turn_debounce = 0

        # PHA 2: CUA TRÁI CUNG TRÊN [01] ──► PHA 3: CUA PHẢI VÀO CHỮ S [02]
        elif self.current_phase == TrackPhase.PHASE_2_TURN_LEFT_UPPER:
            # Nhận diện đảo dấu sang CUA PHẢI (curvature > +0.38)
            if curvature > 0.38 or eff_k > 0.38:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self._switch_phase(TrackPhase.PHASE_3_TURN_RIGHT_S_ENTRY, now)
            else:
                self._turn_debounce = 0

        # PHA 3: CUA PHẢI VÀO CHỮ S [02] ──► PHA 4: CUA TRÁI THOÁT S & ÔM CUNG DƯỚI [03 -> 02 -> 01]
        elif self.current_phase == TrackPhase.PHASE_3_TURN_RIGHT_S_ENTRY:
            # Nhận diện đảo dấu sang CUA TRÁI (curvature < -0.38)
            if curvature < -0.38 or eff_k < -0.38:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self._switch_phase(TrackPhase.PHASE_4_TURN_LEFT_S_EXIT_LOWER, now)
            else:
                self._turn_debounce = 0

        # PHA 4: CUA TRÁI THOÁT S & ÔM CUNG DƯỚI ──► PHA 1: THẲNG XUẤT PHÁT (HOÀN THÀNH 1 VÒNG!)
        elif self.current_phase == TrackPhase.PHASE_4_TURN_LEFT_S_EXIT_LOWER:
            # Chỉ cho phép cán đích khi xe đã qua hết cung dưới (> 2.0s) VÀ đường THẲNG tắp (|k| < 0.25)
            if time_in_phase >= 2.0 and abs(curvature) < 0.25 and abs(heading_error_deg) < 12.0:
                self._turn_debounce += 1
                if self._turn_debounce >= self._debounce_required:
                    self.lap_count += 1
                    self.last_lap_time = now - self.lap_start_time
                    self.lap_start_time = now
                    print(f"🏁 [LAP #{self.lap_count}] Hoàn thành 1 vòng đua trong: {self.last_lap_time:.2f}s!")
                    self._switch_phase(TrackPhase.PHASE_1_STRAIGHT, now)
            else:
                self._turn_debounce = 0

        return self._get_profile(time_in_phase)

    def _switch_phase(self, new_phase: TrackPhase, now: float):
        """Chuyển sang pha tiếp theo."""
        print(f"🚦 [TRACK EVENT] ➔ Chuyển sang: {new_phase.value} (Lap #{self.lap_count + 1})")
        self.current_phase = new_phase
        self.phase_start_time = now
        self._turn_debounce = 0

    def _get_profile(self, time_in_phase: float) -> PhaseControlProfile:
        """Xuất quy tắc lái & ga cho từng pha chính xác."""
        p = self.current_phase

        # 🚀 1. ĐOẠN THẲNG XUẤT PHÁT [05 -> 04]: 100% Ga xé gió, nhìn xa 0.80m, không bù lái ảo
        if p == TrackPhase.PHASE_1_STRAIGHT:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.max_speed,
                lookahead_m=0.80,
                feedforward_gain=0.0,
                feedforward_sign=0.0,
                status_desc="🚀 1. THẲNG 05->04 (100% GA)"
            )

        # 🔄 2. CUA TRÁI CUNG TRÊN [01]: 68% Ga ôm cua mượt, nhìn 0.30m, bù lái Trái 40%
        elif p == TrackPhase.PHASE_2_TURN_LEFT_UPPER:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.cruise_speed,
                lookahead_m=0.30,
                feedforward_gain=0.40,
                feedforward_sign=-1.0,
                status_desc="🔄 2. CUA TRÁI CUNG TRÊN (68% GA)"
            )

        # 🛑 3. CUA PHẢI VÀO CHỮ S [02 -> 03]: 36% Ga an toàn tuyệt đối, nhìn 0.24m, bù lái Phải 45%
        elif p == TrackPhase.PHASE_3_TURN_RIGHT_S_ENTRY:
            return PhaseControlProfile(
                phase=p,
                target_throttle=self.corner_safe_speed,
                lookahead_m=0.24,
                feedforward_gain=0.45,
                feedforward_sign=1.0,
                status_desc="🛑 3. CUA PHẢI VÀO CHỮ S (36% GA)"
            )

        # 🔄 4. CUA TRÁI THOÁT S & ÔM TRỌN CUNG TRÒN DƯỚI [03 -> 02 -> 01 -> 05]:
        # - Trong chữ S (dưới 1.4s): 36% ga an toàn
        # - Ra cung tròn dưới (sau 1.4s): Mở ga lên 68% ôm cua về đích
        # - LUÔN LUÔN BÙ LÁI CUA TRÁI (-45%), KHÔNG BAO GIỜ BẺ PHẢI!
        elif p == TrackPhase.PHASE_4_TURN_LEFT_S_EXIT_LOWER:
            throttle = self.corner_safe_speed if time_in_phase < 1.4 else self.cruise_speed
            lookahead = 0.24 if time_in_phase < 1.4 else 0.32
            return PhaseControlProfile(
                phase=p,
                target_throttle=throttle,
                lookahead_m=lookahead,
                feedforward_gain=0.45,
                feedforward_sign=-1.0,      # LUÔN KHÓA BÙ LÁI CUA TRÁI
                status_desc="🏁 4. CUA TRÁI THOÁT S & CUNG DƯỚI (TRÁI 100%)"
            )

        return PhaseControlProfile(
            phase=TrackPhase.PHASE_1_STRAIGHT,
            target_throttle=self.corner_safe_speed,
            lookahead_m=0.30,
            feedforward_gain=0.30,
            feedforward_sign=0.0,
            status_desc="TRACKING"
        )
