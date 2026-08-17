#!/usr/bin/env python3
"""
V3 / V3.1 Control — Steering Rate Limiter & Saturation

Applies three filters to raw steering commands:
1. Saturation: clamp to [-1, 1]
2. Rate limiting: limit change per frame to max_steer_rate
3. Optional light low-pass filter (EMA)

V3.1 additions (backward compatible):
4. Speed-dependent gain: reduce steering sensitivity at high speed
   to prevent the car from jerking off the track on oval curves.
   When current_speed is not provided, behaves identically to V3.

Prevents sudden steering jumps that cause the car to jerk.
V2 had only a simple 0.7/0.3 blend, which was both too aggressive
and not configurable.

WARNING: Excessive low-pass filtering creates control latency.
The LPF alpha should be close to 1.0 (minimal filtering).
Rate limiting is the primary smoothing mechanism.
"""


class SteeringFilter:
    """Rate-limited and saturated steering output filter."""

    def __init__(self, config):
        """
        Args:
            config: V3Config instance.
        """
        self.max_rate = config.max_steer_rate
        self.lpf_alpha = config.steer_lpf_alpha
        self.prev_output = 0.0

        # V3.1: Speed-adaptive parameters
        self.min_speed = config.min_speed
        self.max_speed = config.max_speed
        self.high_speed_gain = getattr(config, 'high_speed_steer_gain', 1.0)

    def filter(self, raw_steering, current_speed=None):
        """Apply rate limiting, saturation, and optional LPF.

        Args:
            raw_steering: Raw steering command from controller.
            current_speed: (V3.1) Current throttle/speed. None = V3 behavior.

        Returns:
            Filtered steering command in [-1.0, 1.0].
        """
        # Step 0 (V3.1): Speed-dependent gain
        # At low speed (min_speed): gain = 1.0 (full responsiveness for tight turns)
        # At high speed (max_speed): gain = high_speed_gain (smoother, less jerky)
        if current_speed is not None and self.high_speed_gain < 1.0:
            speed_range = self.max_speed - self.min_speed
            if speed_range > 1e-6:
                speed_ratio = max(0.0, min(1.0,
                    (current_speed - self.min_speed) / speed_range))
            else:
                speed_ratio = 0.0
            gain = 1.0 - (1.0 - self.high_speed_gain) * speed_ratio
            raw_steering = raw_steering * gain
        # else: gain = 1.0, no change (V3 behavior)

        # Step 1: Saturation
        saturated = max(-1.0, min(1.0, raw_steering))

        # Step 2: Rate limiting
        # V3.1: Rate limit also reduces at high speed for smoother cornering
        effective_rate = self.max_rate
        if current_speed is not None and self.high_speed_gain < 1.0:
            # At high speed, tighter rate limit (50-100% of max_rate)
            effective_rate = self.max_rate * (0.5 + 0.5 * (1.0 - speed_ratio))

        delta = saturated - self.prev_output
        if abs(delta) > effective_rate:
            delta = effective_rate if delta > 0 else -effective_rate
        rate_limited = self.prev_output + delta

        # Step 3: Optional light LPF
        # alpha=1.0 means no filtering, alpha=0.7 means mild smoothing
        filtered = self.lpf_alpha * rate_limited + (1.0 - self.lpf_alpha) * self.prev_output

        # Final saturation (safety)
        filtered = max(-1.0, min(1.0, filtered))

        self.prev_output = filtered
        return filtered

    def reset(self):
        """Reset filter state (e.g., on state transition)."""
        self.prev_output = 0.0
