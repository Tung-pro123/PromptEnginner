#!/usr/bin/env python3
"""
V3 Perception — Camera Undistortion

Optional module. If camera_matrix and dist_coeffs are provided in config,
applies lens distortion correction. Otherwise passes through unchanged.

Usage:
    undistorter = Undistorter(config)
    corrected = undistorter.process(frame)
"""

import cv2
import numpy as np


class Undistorter:
    """Removes lens distortion using camera intrinsic parameters.

    If calibration is not available (camera_matrix is None), this module
    is a no-op passthrough — no computation wasted.
    """

    def __init__(self, config):
        """
        Args:
            config: V3Config instance with camera_matrix and dist_coeffs.
        """
        self.enabled = (config.camera_matrix is not None
                        and config.dist_coeffs is not None)

        if self.enabled:
            self.camera_matrix = config.camera_matrix
            self.dist_coeffs = config.dist_coeffs
            h, w = config.image_height, config.image_width

            # Pre-compute optimal new camera matrix and undistortion maps
            # alpha=0 crops the result to remove black borders
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), alpha=0
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None,
                self.new_camera_matrix, (w, h), cv2.CV_16SC2
            )

    def process(self, frame):
        """Undistort a single frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            Undistorted BGR image, same size as input.
            If calibration is not available, returns the input unchanged.
        """
        if not self.enabled:
            return frame

        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
