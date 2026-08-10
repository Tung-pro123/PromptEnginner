#!/usr/bin/env python3
"""
V3 Control — Steering Rate Limiter & Saturation

Applies three filters to raw steering commands:
1. Saturation: clamp to [-1, 1]
2. Rate limiting: limit change per frame to max_steer_rate
3. Optional light low-pass filter (EMA)

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

    def filter(self, raw_steering):
        """Apply rate limiting, saturation, and optional LPF.

        Args:
            raw_steering: Raw steering command from controller.

        Returns:
            Filtered steering command in [-1.0, 1.0].
        """
        # Step 1: Saturation
        saturated = max(-1.0, min(1.0, raw_steering))

        # Step 2: Rate limiting
        delta = saturated - self.prev_output
        if abs(delta) > self.max_rate:
            delta = self.max_rate if delta > 0 else -self.max_rate
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
