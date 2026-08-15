#!/usr/bin/env python3
"""
V3 Control — Corridor-Based Local Trajectory Planner

Features:
1. True camera-LiDAR temporal synchronization and missing/static-scan watchdog.
2. LiDAR extrinsic mounting offset compensation (x_lidar, y_lidar).
3. Kinematic Ackermann candidate rollouts spanning the road corridor.
4. Dense Multi-disk vehicle footprint sweeping (Front, Center, Rear) with local segment headings.
5. Strict stopping distance safety enforcement:
       d_stop = v^2 / (2 * a_brake) + v * t_reaction
   Stops immediately if an obstacle is within d_stop and no valid evasion path exists.
6. Boundary freshness gating: disables evasion and safely stops if boundaries remain stale > 0.8s.
7. Active emergency braking with speed_factor = 0.0 when blocked.
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from src.speed_track.control.trajectory import TrajectoryPoint, TrajectoryResult, TrajectoryGenerator, KinematicBicycleModel


@dataclass
class CandidateTrajectory:
    """A single candidate rollout trajectory with safety annotations."""
    points: List[TrajectoryPoint] = field(default_factory=list)
    target_steer_rad: float = 0.0
    lateral_offset_m: float = 0.0
    is_valid: bool = True
    invalid_reason: str = ""
    cost: float = float('inf')
    min_obstacle_dist: float = float('inf')
    trajectory_result: Optional[TrajectoryResult] = None


@dataclass
class CorridorPlannerResult:
    """Output of the corridor trajectory planner."""
    selected_trajectory: Optional[TrajectoryResult] = None
    all_candidates: List[CandidateTrajectory] = field(default_factory=list)
    selected_steer_rad: float = 0.0
    selected_offset_m: float = 0.0
    safe_to_proceed: bool = True
    speed_factor: float = 1.0
    reason: str = "ok"
    d_stop: float = 0.0
    stale_lidar: bool = False


class CorridorPlanner:
    """Local rollout trajectory planner with swept multi-disk footprint checking and active braking."""

    def __init__(self, config, bev_transform, trajectory_generator=None):
        self.cfg = config
        self.bev = bev_transform
        self.traj_gen = trajectory_generator or TrajectoryGenerator(config, bev_transform)
        self.bicycle_model = KinematicBicycleModel(config, bev_transform)

        self.last_selected_steer_rad = 0.0
        self.last_selected_offset = 0.0
        self.has_received_first_scan = False
        self.last_scan_received_time = 0.0
        self._prev_scan_stamp = None
        self.last_valid_plan_time = time.time()

    def get_max_safe_offset(self, boundary_stale: bool = False) -> float:
        """Calculate max allowable lateral evasion offset to stay inside road corridor."""
        cfg = self.cfg
        if boundary_stale:
            return 0.0

        track_w = getattr(cfg, 'track_width_m', 0.60)
        car_w = getattr(cfg, 'car_width_m', 0.18)
        margin = getattr(cfg, 'safety_margin_m', 0.04)
        configured_max = getattr(cfg, 'max_evade_offset_m', 0.18)

        corridor_max = (track_w / 2.0) - (car_w / 2.0) - margin
        return max(0.04, min(configured_max, corridor_max))

    def extract_obstacle_points(self, scan_msg, camera_timestamp: Optional[float] = None, now: Optional[float] = None) -> Tuple[List[Tuple[float, float, float]], bool]:
        """Extract obstacle points in vehicle metric frame (x_m right, y_m forward)."""
        if now is None:
            now = time.time()

        if scan_msg is None:
            return [], False

        cfg = self.cfg
        is_stale = False

        # 1. Camera - LiDAR Temporal Sync & Scan Freshness Check (P0 Fix)
        scan_stamp = None
        if hasattr(scan_msg, 'header') and hasattr(scan_msg.header, 'stamp'):
            try:
                scan_stamp = scan_msg.header.stamp.to_sec() if hasattr(scan_msg.header.stamp, 'to_sec') else float(scan_msg.header.stamp)
            except Exception:
                scan_stamp = None

        if scan_stamp is not None and scan_stamp > 0:
            self.has_received_first_scan = True

            if scan_stamp != self._prev_scan_stamp:
                self.last_scan_received_time = now
                self._prev_scan_stamp = scan_stamp

            if (now - scan_stamp) > getattr(cfg, 'lidar_max_age_s', 0.10):
                is_stale = True

            if camera_timestamp is not None and camera_timestamp > 0:
                skew = abs(camera_timestamp - scan_stamp)
                if skew > getattr(cfg, 'lidar_max_sync_skew_s', 0.10):
                    is_stale = True
        else:
            is_stale = True

        # 2. LiDAR Extrinsics
        lx_offset = getattr(cfg, 'lidar_x_offset_m', 0.0)
        ly_offset = getattr(cfg, 'lidar_y_offset_m', 0.10)

        obs_points = []
        for i, d in enumerate(scan_msg.ranges):
            if not (scan_msg.range_min < d < scan_msg.range_max):
                continue
            if d > cfg.apf_influence_dist:
                continue

            deg = math.degrees(
                scan_msg.angle_min + i * scan_msg.angle_increment
            ) + cfg.lidar_offset_deg
            angle = math.radians((deg + 180) % 360 - 180)
            angle_deg = math.degrees(angle)

            if abs(angle_deg) > 90:
                continue  # only forward semicircle

            # Transform from LiDAR frame to Vehicle frame (+y forward, +x right)
            xm_lidar = -d * math.sin(angle)
            ym_lidar = d * math.cos(angle)

            xm = xm_lidar + lx_offset
            ym = ym_lidar + ly_offset
            dist_veh = math.hypot(xm, ym)

            if 0.0 < ym < 2.0 and abs(xm) < 1.2:
                obs_points.append((xm, ym, dist_veh))

        return obs_points, is_stale

    @staticmethod
    def _dist_point_to_segment(px, py, x1, y1, x2, y2) -> float:
        """Compute shortest Euclidean distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return math.hypot(px - x1, py - y1)

        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def plan(self, lane_state, scan_msg, current_speed: float = 0.0,
             camera_timestamp: Optional[float] = None, current_steer_rad: float = 0.0) -> CorridorPlannerResult:
        """Plan the optimal collision-free trajectory using Control-Lattice Kinematic Bicycle Rollouts.

        Args:
            lane_state: LaneState with centerline, left, and right boundary polynomials.
            scan_msg: sensor_msgs/LaserScan or None.
            current_speed: dynamic vehicle speed in m/s.
            camera_timestamp: timestamp of camera frame for sync.
            current_steer_rad: current front wheel steering angle (rad).

        Returns:
            CorridorPlannerResult with selected trajectory, optimal steering, and safety decision.
        """
        cfg = self.cfg
        now = time.time()

        if lane_state.centerline_poly is None:
            return CorridorPlannerResult(safe_to_proceed=False, speed_factor=0.0, reason="no_lane")

        # 1. Dynamic Stopping Distance Calculation
        a_brake = getattr(cfg, 'a_brake_max', 2.5)
        t_react = getattr(cfg, 't_reaction_s', 0.10)
        v = max(0.0, current_speed)
        d_stop = (v ** 2) / (2.0 * a_brake) + v * t_react
        margin = getattr(cfg, 'safety_margin_m', 0.04)

        # 2. Extract Obstacle Pointcloud with Extrinsics & Sync Check
        obs_points, is_stale = self.extract_obstacle_points(scan_msg, camera_timestamp, now)

        # 3. LiDAR Watchdog: Check if LiDAR is completely missing or timed out
        lidar_timeout = getattr(cfg, 'lidar_timeout_s', 0.80)

        if not self.has_received_first_scan:
            fallback_traj = self.traj_gen.generate(lane_state, current_speed=0.0, lateral_offset_m=0.0)
            return CorridorPlannerResult(
                selected_trajectory=fallback_traj,
                safe_to_proceed=False,
                speed_factor=0.0,
                reason="waiting_for_first_lidar_scan",
                d_stop=d_stop,
                stale_lidar=True
            )

        time_since_scan = now - self.last_scan_received_time
        if scan_msg is None or time_since_scan > lidar_timeout:
            fallback_traj = self.traj_gen.generate(lane_state, current_speed=0.0, lateral_offset_m=0.0)
            return CorridorPlannerResult(
                selected_trajectory=fallback_traj,
                safe_to_proceed=False,
                speed_factor=0.0,
                reason="lidar_watchdog_timeout",
                d_stop=d_stop,
                stale_lidar=True
            )

        # 4. Boundary Freshness & Timeout Check
        boundary_timeout = getattr(cfg, 'boundary_stale_timeout_s', 0.40)
        boundary_stop_timeout = getattr(cfg, 'boundary_stop_timeout_s', 0.80)
        left_age = now - getattr(lane_state, 'left_poly_timestamp', 0.0)
        right_age = now - getattr(lane_state, 'right_poly_timestamp', 0.0)
        left_stale = left_age > boundary_timeout
        right_stale = right_age > boundary_timeout
        boundary_stale = left_stale or right_stale

        if left_age > boundary_stop_timeout and right_age > boundary_stop_timeout:
            fallback_traj = self.traj_gen.generate(lane_state, current_speed=0.0, lateral_offset_m=0.0)
            return CorridorPlannerResult(
                selected_trajectory=fallback_traj,
                safe_to_proceed=False,
                speed_factor=0.0,
                reason="stale_boundaries_timeout",
                d_stop=d_stop,
                stale_lidar=is_stale
            )

        # 5. Generate Candidate Steering Rollouts (Control-Lattice Kinematic Bicycle)
        n_candidates = getattr(cfg, 'n_candidate_rollouts', getattr(cfg, 'n_candidate_trajectories', 11))
        max_steer = getattr(cfg, 'max_steer_rad', 0.436)

        if boundary_stale:
            delta_targets = np.linspace(-max_steer * 0.4, max_steer * 0.4, min(5, n_candidates))
        else:
            delta_targets = np.linspace(-max_steer, max_steer, n_candidates)

        # Ideal feedforward steering for lane centerline
        kappa_lane = self.bev.curvature_px_to_metric(lane_state.centerline_poly)(float(cfg.image_height))
        delta_lane_ff = math.atan(self.bicycle_model.wheelbase * kappa_lane)
        delta_lane_ff = max(-max_steer, min(max_steer, delta_lane_ff))

        target_list = list(delta_targets)
        target_list.append(delta_lane_ff)
        target_list = sorted(list(set(target_list)))

        car_w = getattr(cfg, 'car_width_m', 0.18)
        car_l = getattr(cfg, 'car_length_m', 0.25)
        r_body = car_w / 2.0 + margin
        l_half = car_l / 2.0

        left_poly = lane_state.left_poly
        right_poly = lane_state.right_poly
        lane_w_px = cfg.expected_lane_width_m * cfg.px_per_meter_x

        candidates: List[CandidateTrajectory] = []
        horizon_s = getattr(cfg, 'rollout_horizon_s', 0.90)
        dt_s = getattr(cfg, 'rollout_dt_s', 0.03)

        for delta_target in target_list:
            cand = CandidateTrajectory(target_steer_rad=float(delta_target))
            cand.points = self.bicycle_model.simulate_rollout(
                delta_target=delta_target,
                current_steer_rad=current_steer_rad,
                speed_m_s=v,
                horizon_s=horizon_s,
                dt_s=dt_s
            )

            if not cand.points or len(cand.points) < 2:
                cand.is_valid = False
                cand.invalid_reason = "no_points"
                candidates.append(cand)
                continue

            # Check 1: Multi-Disk Swept Footprint Corridor Containment
            corridor_breached = False
            for i in range(len(cand.points) - 1):
                p1 = cand.points[i]
                p2 = cand.points[i + 1]
                dx = p2.x_m - p1.x_m
                dy = p2.y_m - p1.y_m
                seg_heading = math.atan2(dx, dy) if abs(dy) > 1e-4 else p1.yaw_rad

                for t in (0.0, 0.5, 1.0):
                    px = p1.x_m + t * dx
                    py = p1.y_m + t * dy

                    disks = [
                        (px + l_half * math.sin(seg_heading), py + l_half * math.cos(seg_heading)),
                        (px, py),
                        (px - l_half * math.sin(seg_heading), py - l_half * math.cos(seg_heading)),
                    ]

                    for (dx_m, dy_m) in disks:
                        _, y_px_disk = self.bev.metric_to_px(dx_m, dy_m)
                        if 0 <= y_px_disk <= cfg.image_height:
                            # Check Left Boundary
                            if left_poly is not None and not left_stale:
                                x_l_px = np.polyval(left_poly, y_px_disk)
                                x_l_m, _ = self.bev.px_to_metric(x_l_px, y_px_disk)
                                if (dx_m - r_body) < x_l_m:
                                    corridor_breached = True
                                    break
                            elif right_poly is not None and not right_stale:
                                x_r_px = np.polyval(right_poly, y_px_disk)
                                x_l_px = x_r_px - lane_w_px
                                x_l_m, _ = self.bev.px_to_metric(x_l_px, y_px_disk)
                                if (dx_m - r_body) < x_l_m:
                                    corridor_breached = True
                                    break

                            # Check Right Boundary
                            if right_poly is not None and not right_stale:
                                x_r_px = np.polyval(right_poly, y_px_disk)
                                x_r_m, _ = self.bev.px_to_metric(x_r_px, y_px_disk)
                                if (dx_m + r_body) > x_r_m:
                                    corridor_breached = True
                                    break
                            elif left_poly is not None and not left_stale:
                                x_l_px = np.polyval(left_poly, y_px_disk)
                                x_r_px = x_l_px + lane_w_px
                                x_r_m, _ = self.bev.px_to_metric(x_r_px, y_px_disk)
                                if (dx_m + r_body) > x_r_m:
                                    corridor_breached = True
                                    break

                    if corridor_breached:
                        break
                if corridor_breached:
                    break

            if corridor_breached:
                cand.is_valid = False
                cand.invalid_reason = "corridor_breach"
                candidates.append(cand)
                continue

            # Check 2: Multi-Disk Swept Segment Collision & Longitudinal Arc Stopping Distance
            min_dist_to_obs = float('inf')
            collision_detected = False

            arc_lengths = [0.0]
            for j in range(1, len(cand.points)):
                seg_d = math.hypot(cand.points[j].x_m - cand.points[j - 1].x_m,
                                   cand.points[j].y_m - cand.points[j - 1].y_m)
                arc_lengths.append(arc_lengths[-1] + seg_d)

            for (ox, oy, odist) in obs_points:
                for i in range(len(cand.points) - 1):
                    p1 = cand.points[i]
                    p2 = cand.points[i + 1]
                    dx = p2.x_m - p1.x_m
                    dy = p2.y_m - p1.y_m
                    seg_len = math.hypot(dx, dy)
                    seg_heading = math.atan2(dx, dy) if abs(dy) > 1e-4 else p1.yaw_rad

                    d_center_seg = self._dist_point_to_segment(ox, oy, p1.x_m, p1.y_m, p2.x_m, p2.y_m)
                    min_dist_to_obs = min(min_dist_to_obs, d_center_seg)

                    if seg_len > 1e-6:
                        t = max(0.0, min(1.0, ((ox - p1.x_m) * dx + (oy - p1.y_m) * dy) / (seg_len * seg_len)))
                    else:
                        t = 0.0
                    arc_dist_to_obs = arc_lengths[i] + t * seg_len

                    f1_x = p1.x_m + l_half * math.sin(seg_heading)
                    f1_y = p1.y_m + l_half * math.cos(seg_heading)
                    f2_x = p2.x_m + l_half * math.sin(seg_heading)
                    f2_y = p2.y_m + l_half * math.cos(seg_heading)
                    d_front_seg = self._dist_point_to_segment(ox, oy, f1_x, f1_y, f2_x, f2_y)
                    min_dist_to_obs = min(min_dist_to_obs, d_front_seg)

                    r1_x = p1.x_m - l_half * math.sin(seg_heading)
                    r1_y = p1.y_m - l_half * math.cos(seg_heading)
                    r2_x = p2.x_m - l_half * math.sin(seg_heading)
                    r2_y = p2.y_m - l_half * math.cos(seg_heading)
                    d_rear_seg = self._dist_point_to_segment(ox, oy, r1_x, r1_y, r2_x, r2_y)
                    min_dist_to_obs = min(min_dist_to_obs, d_rear_seg)

                    min_swept_dist = min(d_center_seg, d_front_seg, d_rear_seg)

                    if min_swept_dist < r_body:
                        collision_detected = True
                        break

                    if min_swept_dist < (r_body + 0.05) and arc_dist_to_obs <= (d_stop + margin):
                        collision_detected = True
                        break

                if collision_detected:
                    break

            cand.min_obstacle_dist = min_dist_to_obs

            if collision_detected:
                cand.is_valid = False
                cand.invalid_reason = "obstacle_collision"
                candidates.append(cand)
                continue

            # Check 3: Candidate Cost Scoring (Lane Reference Cost Function)
            w_lane = getattr(cfg, 'w_lane', 1.0)
            w_steer = getattr(cfg, 'w_steer', 0.25)
            w_rate = getattr(cfg, 'w_rate', 0.35)
            w_clearance = getattr(cfg, 'w_clearance', 0.80)
            w_progress = getattr(cfg, 'w_progress', 0.20)

            # Measure lateral deviation from reference lane
            lane_errors = []
            for p in cand.points:
                _, y_px = self.bev.metric_to_px(p.x_m, p.y_m)
                if 0 <= y_px <= cfg.image_height:
                    x_ref_px = np.polyval(lane_state.centerline_poly, y_px)
                    x_ref_m, _ = self.bev.px_to_metric(x_ref_px, y_px)
                    lane_errors.append((p.x_m - x_ref_m) ** 2)

            lane_error_cost = math.sqrt(np.mean(lane_errors)) if lane_errors else 0.0
            steer_cost = abs(delta_target - delta_lane_ff)
            rate_cost = abs(delta_target - current_steer_rad)
            
            # Non-linear clearance penalty: heavily penalizes passing close to obstacles
            clear_margin = min_dist_to_obs - r_body
            if clear_margin < 0.25 and min_dist_to_obs < float('inf'):
                clearance_penalty = ((0.25 - max(0.0, clear_margin)) / 0.25) ** 2 * 2.5
            else:
                clearance_penalty = 0.0
                
            progress_reward = cand.points[-1].y_m

            cand.cost = (
                w_lane * lane_error_cost +
                w_steer * steer_cost +
                w_rate * rate_cost +
                w_clearance * clearance_penalty -
                w_progress * progress_reward
            )
            cand.is_valid = True

            # Package TrajectoryResult for controller / visualizer
            Ld = self.traj_gen._adaptive_lookahead(v, kappa_lane)
            target_pt = self.traj_gen._select_target(cand.points, Ld)

            # True physical maximum curvature along the rollout trajectory (accounting for servo ramp)
            curvatures_rollout = [
                abs(math.tan(p.steer_rad) / self.bicycle_model.wheelbase)
                for p in cand.points
            ]
            kappa_max_rollout = max(curvatures_rollout) if curvatures_rollout else 0.0

            cand.trajectory_result = TrajectoryResult(
                points=cand.points,
                target=target_pt,
                lookahead_m=Ld,
                curvature=kappa_max_rollout,
                heading_error=cand.points[0].yaw_rad,
                lateral_error_m=cand.points[0].x_m
            )

            # Lateral offset of candidate trajectory at horizon endpoint
            x_ref_end_px = np.polyval(lane_state.centerline_poly, cand.points[-1].y_px)
            x_ref_end_m, _ = self.bev.px_to_metric(x_ref_end_px, cand.points[-1].y_px)
            cand.lateral_offset_m = cand.points[-1].x_m - x_ref_end_m

            candidates.append(cand)

        # 6. Optimal Candidate Selection & Active Emergency Braking
        valid_candidates = [c for c in candidates if c.is_valid]

        if not valid_candidates:
            fallback_traj = self.traj_gen.generate(lane_state, current_speed=0.0, lateral_offset_m=0.0)
            return CorridorPlannerResult(
                selected_trajectory=fallback_traj,
                all_candidates=candidates,
                selected_steer_rad=0.0,
                selected_offset_m=0.0,
                safe_to_proceed=False,
                speed_factor=0.0,
                reason="corridor_blocked_safe_stop",
                d_stop=d_stop,
                stale_lidar=is_stale
            )

        best_cand = min(valid_candidates, key=lambda c: c.cost)
        self.last_selected_steer_rad = best_cand.target_steer_rad
        self.last_selected_offset = best_cand.lateral_offset_m
        self.last_valid_plan_time = now

        speed_factor = 1.0
        if is_stale or scan_msg is None:
            speed_factor = min(speed_factor, 0.5)
        if boundary_stale:
            speed_factor = min(speed_factor, getattr(cfg, 'crawl_speed', 0.12) / cfg.cruise_speed)
        elif best_cand.min_obstacle_dist < 0.60:
            speed_factor = min(speed_factor, max(0.4, best_cand.min_obstacle_dist / 0.60))

        return CorridorPlannerResult(
            selected_trajectory=best_cand.trajectory_result,
            all_candidates=candidates,
            selected_steer_rad=best_cand.target_steer_rad,
            selected_offset_m=best_cand.lateral_offset_m,
            safe_to_proceed=True,
            speed_factor=speed_factor,
            reason="optimal_path_found",
            d_stop=d_stop,
            stale_lidar=is_stale
        )
