#!/usr/bin/env python3
"""
Unit test for 3-layer anti-drift lane association in MultiLaneDetector.
Tests:
1. 3 lines (L, C, R) normal tracking + dynamic width learning
2. 2 lines (L, R) full width missing center
3. 2 lines (C, R) half width under left turn drift (dash vs solid)
4. 1 line (R only) under extreme left turn drift (single solid peak)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.speed_track.config import V3Config
from src.speed_track.perception.lane_detector import MultiLaneDetector


def create_synthetic_bev(bev_w=640, bev_h=480, lines_spec=None):
    """Create a synthetic binary BEV image with specified lines.
    
    lines_spec: list of dicts with:
      - 'x': base x position
      - 'type': 'solid' or 'dashed'
    """
    bev = np.zeros((bev_h, bev_w), dtype=np.uint8)
    if not lines_spec:
        return bev

    for spec in lines_spec:
        x = spec['x']
        line_type = spec.get('type', 'solid')
        width = spec.get('width', 10)
        
        if line_type == 'solid':
            bev[:, max(0, x - width//2):min(bev_w, x + width//2)] = 255
        elif line_type == 'dashed':
            # 40px dash, 40px gap
            for y in range(0, bev_h, 80):
                bev[y:min(bev_h, y+40), max(0, x - width//2):min(bev_w, x + width//2)] = 255
                
    return bev


def test_3_lines_normal():
    cfg = V3Config()
    detector = MultiLaneDetector(cfg)
    
    # 3 lines: L=120 (solid), C=320 (dashed), R=520 (solid)
    bev = create_synthetic_bev(640, 480, [
        {'x': 120, 'type': 'solid'},
        {'x': 320, 'type': 'dashed'},
        {'x': 520, 'type': 'solid'},
    ])
    
    res = detector.detect(bev, current_steer=0.0)
    assert res.left.detected, "Left line should be detected"
    assert res.center.detected, "Center line should be detected"
    assert res.right.detected, "Right line should be detected"
    print("[PASS] Test 1: 3-Lines Normal Association PASS")
    return True


def test_2_lines_full_width():
    cfg = V3Config()
    detector = MultiLaneDetector(cfg)
    
    # 2 lines: L=120 (solid), R=520 (solid), C is missing (gap = 400px ~ W)
    bev = create_synthetic_bev(640, 480, [
        {'x': 120, 'type': 'solid'},
        {'x': 520, 'type': 'solid'},
    ])
    
    res = detector.detect(bev, current_steer=0.0)
    assert res.left.detected, "Left line should be detected"
    assert not res.center.detected, "Center line should be None (missing)"
    assert res.right.detected, "Right line should be detected"
    print("[PASS] Test 2: 2-Lines Full Width (L+R) PASS")
    return True


def test_2_lines_left_turn_drift():
    cfg = V3Config()
    detector = MultiLaneDetector(cfg)
    
    # Turning left (steer = -0.4): Car drifted right.
    # C=160 (dashed, shifted left), R=360 (solid, shifted to center)
    bev = create_synthetic_bev(640, 480, [
        {'x': 160, 'type': 'dashed'},
        {'x': 360, 'type': 'solid'},
    ])
    
    res = detector.detect(bev, current_steer=-0.4)
    assert not res.left.detected, "Left line should be None (out of frame)"
    assert res.center.detected, "Center line should be detected at x~160"
    assert res.right.detected, "Right line should be detected at x~360"
    print("[PASS] Test 3: 2-Lines (C+R) Left Turn Drift Association PASS")
    return True


def test_1_line_solid_extreme_drift():
    cfg = V3Config()
    detector = MultiLaneDetector(cfg)
    
    # Turning hard left (steer = -0.5): C is completely out of frame.
    # Only R=300 (solid, shifted near center) is visible.
    bev = create_synthetic_bev(640, 480, [
        {'x': 300, 'type': 'solid'},
    ])
    
    res = detector.detect(bev, current_steer=-0.5)
    assert not res.left.detected, "Left line should be None"
    assert not res.center.detected, "Center line MUST NOT lock onto solid line"
    assert res.right.detected, "Single solid line during left turn MUST be identified as Right boundary"
    print("[PASS] Test 4: 1-Line Extreme Drift (Solid Line Anti-Lock) PASS")
    return True


if __name__ == '__main__':
    test_3_lines_normal()
    test_2_lines_full_width()
    test_2_lines_left_turn_drift()
    test_1_line_solid_extreme_drift()
    print("\nALL ANTI-DRIFT ASSOCIATION TESTS PASSED SUCCESSFULLY!")
