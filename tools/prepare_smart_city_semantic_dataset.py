#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a clean six-class detection dataset from the team Roboflow ZIP.

The source export mixes five-column YOLO boxes and segmentation polygons in
the same label files. A detect task must not receive that mixed representation:
this script keeps existing boxes, converts semantic polygons to tight boxes,
removes geometry/action-region classes and writes only five-column rows.
"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import zipfile


SOURCE_NAMES = {
    0: "Corner",
    1: "Decision",
    2: "Forbidden",
    3: "Green_Light",
    4: "Interact",
    5: "Left",
    6: "Red_Light",
    7: "Right",
    8: "straight",
}

SOURCE_TO_SEMANTIC = {
    2: 0,  # Forbidden
    3: 1,  # Green_Light
    5: 2,  # Left
    6: 3,  # Red_Light
    7: 4,  # Right
    8: 5,  # straight
}

SEMANTIC_NAMES = {
    0: "Forbidden",
    1: "Green_Light",
    2: "Left",
    3: "Red_Light",
    4: "Right",
    5: "straight",
}

RARE_CLASS_IDS = frozenset((1, 2, 3))
LIGHT_CLASS_IDS = frozenset((1, 3))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("label coordinate must be finite")
    return value


def _clip(value):
    return min(1.0, max(0.0, float(value)))


def _corners_to_xywh(x1, y1, x2, y2):
    x1, y1, x2, y2 = [_clip(value) for value in (x1, y1, x2, y2)]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("label box has no positive area")
    return (
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5,
        x2 - x1,
        y2 - y1,
    )


def convert_label_line(line):
    """Return ``(new_class_id, x, y, w, h)`` or None for geometry classes."""

    parts = str(line).strip().split()
    if not parts:
        return None
    try:
        source_class = int(parts[0])
    except (TypeError, ValueError):
        raise ValueError("class id must be an integer")
    if source_class not in SOURCE_NAMES:
        raise ValueError("unknown source class id: %s" % source_class)
    if source_class not in SOURCE_TO_SEMANTIC:
        return None

    coordinates = [_finite(value) for value in parts[1:]]
    if len(parts) == 5:
        centre_x, centre_y, width, height = coordinates
        if width <= 0.0 or height <= 0.0:
            raise ValueError("label box has no positive area")
        converted = _corners_to_xywh(
            centre_x - width * 0.5,
            centre_y - height * 0.5,
            centre_x + width * 0.5,
            centre_y + height * 0.5,
        )
    else:
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError("polygon needs at least three x/y points")
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        converted = _corners_to_xywh(min(xs), min(ys), max(xs), max(ys))
    return (SOURCE_TO_SEMANTIC[source_class],) + converted


def convert_label_text(text):
    converted = []
    for line_number, line in enumerate(str(text).splitlines(), 1):
        try:
            row = convert_label_line(line)
        except ValueError as exc:
            raise ValueError("line %d: %s" % (line_number, exc))
        if row is not None:
            converted.append(row)
    return converted


def format_label_rows(rows):
    return "".join(
        "%d %.8f %.8f %.8f %.8f\n" % row for row in rows
    )


def transform_rows_to_crop(rows, crop_box, minimum_retained=0.75):
    """Map normalized detection rows into a normalized crop.

    Objects whose centre is outside the crop or that are mostly clipped are
    removed. Keeping this operation independent of image I/O makes it easy to
    regression-test the label geometry.
    """

    crop_left, crop_top, crop_right, crop_bottom = [
        _finite(value) for value in crop_box
    ]
    if not (
            0.0 <= crop_left < crop_right <= 1.0
            and 0.0 <= crop_top < crop_bottom <= 1.0
    ):
        raise ValueError("crop box must be inside normalized image bounds")

    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    transformed = []
    for row in rows:
        class_id, centre_x, centre_y, width, height = row
        centre_x = _finite(centre_x)
        centre_y = _finite(centre_y)
        width = _finite(width)
        height = _finite(height)
        if width <= 0.0 or height <= 0.0:
            continue
        if not (
                crop_left <= centre_x <= crop_right
                and crop_top <= centre_y <= crop_bottom
        ):
            continue

        left = centre_x - width * 0.5
        top = centre_y - height * 0.5
        right = centre_x + width * 0.5
        bottom = centre_y + height * 0.5
        clipped_left = max(left, crop_left)
        clipped_top = max(top, crop_top)
        clipped_right = min(right, crop_right)
        clipped_bottom = min(bottom, crop_bottom)
        clipped_width = clipped_right - clipped_left
        clipped_height = clipped_bottom - clipped_top
        if clipped_width <= 0.0 or clipped_height <= 0.0:
            continue
        retained = (clipped_width * clipped_height) / (width * height)
        if retained < float(minimum_retained):
            continue

        transformed.append((
            int(class_id),
            ((clipped_left + clipped_right) * 0.5 - crop_left) / crop_width,
            ((clipped_top + clipped_bottom) * 0.5 - crop_top) / crop_height,
            clipped_width / crop_width,
            clipped_height / crop_height,
        ))
    return transformed


def _write_image_crop(source_path, target_path, crop_box):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for light crop augmentation: %s" % exc)

    image = cv2.imread(source_path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("cannot decode training image: %s" % source_path)
    height, width = image.shape[:2]
    left = max(0, min(width - 1, int(round(crop_box[0] * width))))
    top = max(0, min(height - 1, int(round(crop_box[1] * height))))
    right = max(left + 1, min(width, int(round(crop_box[2] * width))))
    bottom = max(top + 1, min(height, int(round(crop_box[3] * height))))
    if not cv2.imwrite(target_path, image[top:bottom, left:right]):
        raise IOError("cannot write cropped training image: %s" % target_path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_data_yaml(output_root):
    yaml_path = os.path.join(output_root, "data.yaml")
    absolute_root = os.path.abspath(output_root).replace("\\", "/")
    lines = [
        "path: %s" % absolute_root,
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for class_id in sorted(SEMANTIC_NAMES):
        lines.append("  %d: %s" % (class_id, SEMANTIC_NAMES[class_id]))
    with open(yaml_path, "w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines) + "\n")
    return yaml_path


def prepare_dataset(
        zip_path, output_root, oversample_min_count=120,
        light_crop_bottom=0.58
):
    zip_path = os.path.abspath(zip_path)
    output_root = os.path.abspath(output_root)
    if not os.path.isfile(zip_path):
        raise IOError("dataset ZIP does not exist: %s" % zip_path)
    if os.path.exists(output_root):
        raise IOError("output already exists; choose a new/empty path")

    samples = {"train": [], "val": [], "test": []}
    original_counts = {
        split: collections.Counter() for split in samples
    }
    source_format_counts = collections.Counter()
    mixed_files = 0
    split_map = {"train": "train", "valid": "val", "test": "test"}

    with zipfile.ZipFile(zip_path) as bundle:
        names = set(bundle.namelist())
        for source_split, target_split in split_map.items():
            prefix = source_split + "/images/"
            image_entries = sorted(
                name for name in names
                if name.startswith(prefix)
                and name.lower().endswith(IMAGE_EXTENSIONS)
            )
            image_dir = os.path.join(output_root, "images", target_split)
            label_dir = os.path.join(output_root, "labels", target_split)
            os.makedirs(image_dir)
            os.makedirs(label_dir)

            for image_entry in image_entries:
                filename = image_entry.rsplit("/", 1)[-1]
                stem = os.path.splitext(filename)[0]
                label_entry = "%s/labels/%s.txt" % (source_split, stem)
                raw_text = (
                    bundle.read(label_entry).decode("utf-8")
                    if label_entry in names else ""
                )
                row_kinds = set()
                for raw_line in raw_text.splitlines():
                    count = len(raw_line.split())
                    if not count:
                        continue
                    kind = "box" if count == 5 else "polygon"
                    source_format_counts[kind] += 1
                    row_kinds.add(kind)
                if len(row_kinds) > 1:
                    mixed_files += 1

                rows = convert_label_text(raw_text)
                for row in rows:
                    original_counts[target_split][row[0]] += 1

                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, stem + ".txt")
                with open(image_path, "wb") as output:
                    output.write(bundle.read(image_entry))
                with open(label_path, "w", encoding="utf-8", newline="\n") as output:
                    output.write(format_label_rows(rows))
                samples[target_split].append((image_path, label_path, rows))

    augmentation_counts = collections.Counter()
    crop_bottom = float(light_crop_bottom)
    if crop_bottom:
        if not 0.50 < crop_bottom <= 1.0:
            raise ValueError("light_crop_bottom must be zero or in (0.50, 1.0]")
        crop_box = (0.0, 0.0, 1.0, crop_bottom)
        for image_path, label_path, rows in list(samples["train"]):
            present = set(row[0] for row in rows)
            if not present.intersection(LIGHT_CLASS_IDS):
                continue
            cropped_rows = transform_rows_to_crop(rows, crop_box)
            if not set(row[0] for row in cropped_rows).intersection(
                    LIGHT_CLASS_IDS
            ):
                continue
            image_stem, image_extension = os.path.splitext(image_path)
            label_stem, label_extension = os.path.splitext(label_path)
            cropped_image = image_stem + "__upper_roi" + image_extension
            cropped_label = label_stem + "__upper_roi" + label_extension
            _write_image_crop(image_path, cropped_image, crop_box)
            with open(
                    cropped_label, "w", encoding="utf-8", newline="\n"
            ) as output:
                output.write(format_label_rows(cropped_rows))
            samples["train"].append(
                (cropped_image, cropped_label, cropped_rows)
            )
            for row in cropped_rows:
                augmentation_counts[row[0]] += 1

    augmented_counts = original_counts["train"] + augmentation_counts
    oversample_factors = {}
    minimum = max(0, int(oversample_min_count))
    for class_id in RARE_CLASS_IDS:
        count = augmented_counts[class_id]
        oversample_factors[class_id] = (
            max(1, int(math.ceil(float(minimum) / count))) if count else 1
        )

    for image_path, label_path, rows in list(samples["train"]):
        present = set(row[0] for row in rows).intersection(RARE_CLASS_IDS)
        repeat_factor = max(
            [oversample_factors[class_id] for class_id in present] or [1]
        )
        image_stem, image_extension = os.path.splitext(image_path)
        label_stem, label_extension = os.path.splitext(label_path)
        for repeat_index in range(1, repeat_factor):
            repeated_image = "%s__repeat_%d%s" % (
                image_stem, repeat_index, image_extension
            )
            repeated_label = "%s__repeat_%d%s" % (
                label_stem, repeat_index, label_extension
            )
            shutil.copyfile(image_path, repeated_image)
            shutil.copyfile(label_path, repeated_label)

    final_counts = {split: collections.Counter() for split in samples}
    final_images = {}
    for split in samples:
        label_dir = os.path.join(output_root, "labels", split)
        image_dir = os.path.join(output_root, "images", split)
        final_images[split] = len(os.listdir(image_dir))
        for filename in os.listdir(label_dir):
            with open(
                    os.path.join(label_dir, filename), "r", encoding="utf-8"
            ) as source:
                for line in source:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(
                            "generated label is not five-column detection: %s"
                            % filename
                        )
                    final_counts[split][int(parts[0])] += 1

    yaml_path = _write_data_yaml(output_root)
    manifest = {
        "source_zip": zip_path,
        "source_sha256": _sha256(zip_path),
        "source_formats": dict(source_format_counts),
        "mixed_label_files": mixed_files,
        "semantic_names": SEMANTIC_NAMES,
        "original_instance_counts": {
            split: dict(counts) for split, counts in original_counts.items()
        },
        "light_crop_bottom": crop_bottom,
        "light_crop_instance_counts": dict(augmentation_counts),
        "pre_oversample_train_counts": dict(augmented_counts),
        "oversample_factors": oversample_factors,
        "final_image_counts": final_images,
        "final_instance_counts": {
            split: dict(counts) for split, counts in final_counts.items()
        },
    }
    with open(
            os.path.join(output_root, "manifest.json"),
            "w", encoding="utf-8"
    ) as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return yaml_path, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare corrected Smart City semantic detection dataset"
    )
    parser.add_argument("zip_path")
    parser.add_argument("output_root")
    parser.add_argument("--oversample-min-count", type=int, default=120)
    parser.add_argument(
        "--light-crop-bottom", type=float, default=0.58,
        help="bottom of full-width upper light crop; pass 0 to disable",
    )
    args = parser.parse_args(argv)
    yaml_path, manifest = prepare_dataset(
        args.zip_path, args.output_root, args.oversample_min_count,
        args.light_crop_bottom
    )
    print("data_yaml=%s" % yaml_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
