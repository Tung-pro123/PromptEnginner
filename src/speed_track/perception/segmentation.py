#!/usr/bin/env python3
"""
V3 Perception — HSV Color Segmentation

Detects red/orange lane markings on the track surface.
Handles hue wrap-around (red spans both ends of HSV hue spectrum).
Uses only MORPH_OPEN to remove noise — does NOT use MORPH_CLOSE,
which would connect the dashed center line into a continuous blob.

Indoor-lighting adaptations (patched for actual camera):
- Gray-world white balance stabilizes hue under fluorescent lights.
- LAB a-channel constraint rejects neutral-gray floor reflections
  that accidentally fall within the HSV hue range.
"""

import cv2
import numpy as np


class ColorSegmenter:
    """HSV-based detection of red/orange lane markings.

    The track has red/orange lines. In HSV, red hue wraps around:
    - Low hue range:  [0, ~20]
    - High hue range: [~155, 180]

    Both ranges share the same S and V minimums.

    Additional constraints for indoor reflective floors:
    - Gray-world white balance normalizes color cast from lighting.
    - LAB a-channel check ensures detected pixels are genuinely red/orange,
      not gray/white floor reflections.
    """

    def __init__(self, config):
        """
        Args:
            config: V3Config instance with HSV parameters.
        """
        self.cfg = config

        self.cfg = config
        
        # We will build lower/upper bounds dynamically in process() 
        # so that Live Calibrator updates take effect instantly!

        # Morphology kernel — small, OPEN only
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (config.morph_kernel_size, config.morph_kernel_size)
        )
        self.morph_iterations = config.morph_iterations

        # CLAHE for lighting robustness
        self.use_clahe = config.use_clahe
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=(config.clahe_tile_size, config.clahe_tile_size)
            )

        # Indoor-lighting features
        self.use_white_balance = config.use_white_balance
        self.use_lab_constraint = config.use_lab_constraint
        self.lab_a_min = config.lab_a_min

    def process(self, frame):
        """Segment lane markings from a BGR frame.

        Pipeline:
            1. (Optional) Gray-world white balance
            2. (Optional) CLAHE on L channel for lighting normalization
               — also extracts LAB a-channel for step 6
            3. Convert to HSV
            4. Threshold two hue ranges
            5. OR the masks
            6. (Optional) AND with LAB a-channel constraint
            7. Morphological OPEN (remove noise dots)

        Args:
            frame: BGR image (numpy array).

        Returns:
            Binary mask (uint8, 0 or 255) where white = detected marking.
        """
        # Step 1: Gray-world white balance
        # Normalizes color cast from indoor fluorescent lighting.
        # Cost: one mean() + one multiply — negligible on Jetson Nano.
        if self.use_white_balance:
            frame = self._gray_world_wb(frame)

        # Step 2: CLAHE preprocessing + extract LAB a-channel
        a_channel_lab = None
        if self.use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = self.clahe.apply(l_channel)

            # Save a_channel for LAB constraint (free — already computed)
            if self.use_lab_constraint:
                a_channel_lab = a_channel

            lab = cv2.merge((l_channel, a_channel, b_channel))
            frame_preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            frame_preprocessed = frame
            # If CLAHE is off but LAB constraint is on, do a separate conversion
            if self.use_lab_constraint:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                a_channel_lab = lab[:, :, 1]

        # Step 3: Convert to HSV
        hsv = cv2.cvtColor(frame_preprocessed, cv2.COLOR_BGR2HSV)

        # Build lower/upper bounds dynamically (For Live Calibrator)
        cfg = self.cfg
        s_max = getattr(cfg, 'hsv_s_max', 255)
        v_max = getattr(cfg, 'hsv_v_max', 255)
        
        lower1_near = np.array([cfg.hsv_h1_min, cfg.hsv_s_min, cfg.hsv_v_min])
        lower2_near = np.array([cfg.hsv_h2_min, cfg.hsv_s_min, cfg.hsv_v_min])
        
        s_far = getattr(cfg, 'hsv_s_min_far', cfg.hsv_s_min)
        v_far = getattr(cfg, 'hsv_v_min_far', cfg.hsv_v_min)
        lower1_far = np.array([cfg.hsv_h1_min, s_far, v_far])
        lower2_far = np.array([cfg.hsv_h2_min, s_far, v_far])
        
        upper1 = np.array([cfg.hsv_h1_max, s_max, v_max])
        upper2 = np.array([cfg.hsv_h2_max, s_max, v_max])
        
        far_y_ratio = getattr(cfg, 'hsv_far_y_split', 0.45)

        # Step 4-5: Dual-range thresholding (Adaptive Distance-Based)
        
        # 4a: Calculate Strict Mask for Near Zone
        mask1_near = cv2.inRange(hsv, lower1_near, upper1)
        mask2_near = cv2.inRange(hsv, lower2_near, upper2)
        mask_near = cv2.bitwise_or(mask1_near, mask2_near)
        
        # 4b: Calculate Loose Mask for Far Zone
        mask1_far = cv2.inRange(hsv, lower1_far, upper1)
        mask2_far = cv2.inRange(hsv, lower2_far, upper2)
        mask_far = cv2.bitwise_or(mask1_far, mask2_far)
        
        # 4c: Merge them (Top = Far, Bottom = Near)
        split_y = int(hsv.shape[0] * far_y_ratio)
        mask = mask_near.copy()
        mask[:split_y, :] = mask_far[:split_y, :]

        # Step 6: LAB a-channel constraint
        # Rejects neutral gray/white floor reflections that accidentally
        # pass the HSV thresholds. Real red/orange has LAB a > ~135;
        # gray reflections have a ≈ 128 (neutral).
        # Cost: one comparison + one bitwise_and — negligible.
        if self.use_lab_constraint and a_channel_lab is not None:
            lab_mask = (a_channel_lab >= self.lab_a_min).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, lab_mask)

        # Step 7: Morphological OPEN only
        # We re-create the kernel dynamically in case blur_size was changed live
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (cfg.morph_kernel_size, cfg.morph_kernel_size)
        )
        
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel,
            iterations=self.morph_iterations
        )

        return mask

    @staticmethod
    def _gray_world_wb(frame):
        """Gray-world white balance normalization.

        Assumes the average color of the scene should be neutral gray.
        Scales each BGR channel so its mean equals the global mean.
        This corrects color casts from non-daylight illumination (e.g.,
        warm fluorescent or cool LED lighting).

        Cost: ~0.3ms on Jetson Nano for 640x480 — negligible.

        Args:
            frame: BGR image (uint8).

        Returns:
            White-balanced BGR image (uint8).
        """
        # Compute per-channel means
        mean_bgr = frame.mean(axis=(0, 1))  # [mean_B, mean_G, mean_R]
        global_mean = mean_bgr.mean()

        # Avoid division by zero for very dark frames
        scale = np.where(mean_bgr > 1.0, global_mean / mean_bgr, 1.0)

        # Apply scaling — clip to [0, 255]
        balanced = np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return balanced

