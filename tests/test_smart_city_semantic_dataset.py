# -*- coding: utf-8 -*-
"""Regression tests for mixed-label to clean-detection conversion."""

from __future__ import absolute_import

import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.prepare_smart_city_semantic_dataset import (  # noqa: E402
    convert_label_line,
    convert_label_text,
    format_label_rows,
    transform_rows_to_crop,
)


class SemanticDatasetConversionTests(unittest.TestCase):
    def test_existing_box_is_kept_and_class_is_remapped(self):
        row = convert_label_line("3 0.50 0.25 0.10 0.04")
        self.assertEqual(1, row[0])
        self.assertAlmostEqual(0.50, row[1])
        self.assertAlmostEqual(0.25, row[2])
        self.assertAlmostEqual(0.10, row[3])
        self.assertAlmostEqual(0.04, row[4])

    def test_polygon_is_converted_to_tight_box(self):
        row = convert_label_line(
            "2 0.10 0.20 0.40 0.20 0.40 0.60 0.10 0.60"
        )
        self.assertEqual(0, row[0])
        self.assertAlmostEqual(0.25, row[1])
        self.assertAlmostEqual(0.40, row[2])
        self.assertAlmostEqual(0.30, row[3])
        self.assertAlmostEqual(0.40, row[4])

    def test_geometry_classes_are_removed(self):
        for source_class in (0, 1, 4):
            self.assertIsNone(
                convert_label_line("%d 0.5 0.5 0.2 0.2" % source_class)
            )

    def test_every_output_row_has_exactly_five_columns(self):
        rows = convert_label_text(
            "6 0.5 0.5 0.1 0.1\n"
            "7 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n"
            "4 0.5 0.5 0.3 0.3\n"
        )
        output = format_label_rows(rows)
        self.assertEqual(2, len(output.splitlines()))
        self.assertTrue(
            all(len(line.split()) == 5 for line in output.splitlines())
        )

    def test_invalid_polygon_and_non_finite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            convert_label_line("2 0.1 0.2 0.3 0.4 0.5")
        with self.assertRaises(ValueError):
            convert_label_line("3 nan 0.5 0.1 0.1")

    def test_upper_crop_enlarges_light_and_maps_box_coordinates(self):
        rows = [(1, 0.25, 0.30, 0.02, 0.03)]
        cropped = transform_rows_to_crop(rows, (0.0, 0.0, 1.0, 0.60))
        self.assertEqual(1, len(cropped))
        self.assertEqual(1, cropped[0][0])
        self.assertAlmostEqual(0.25, cropped[0][1])
        self.assertAlmostEqual(0.50, cropped[0][2])
        self.assertAlmostEqual(0.02, cropped[0][3])
        self.assertAlmostEqual(0.05, cropped[0][4])

    def test_upper_crop_drops_object_outside_crop(self):
        rows = [(4, 0.50, 0.80, 0.10, 0.10)]
        self.assertEqual(
            [], transform_rows_to_crop(rows, (0.0, 0.0, 1.0, 0.60))
        )


if __name__ == "__main__":
    unittest.main()
