#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the corrected six-class Smart City sign/light detector."""

from __future__ import absolute_import, print_function

import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="corrected data.yaml")
    parser.add_argument(
        "--model", default="models/smart_city_semantic_best.pt",
        help="starting YOLO checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="training/runs")
    parser.add_argument("--name", default="semantic_v2_fixed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    project_dir = os.path.abspath(args.project)
    if not os.path.isfile(args.data):
        raise SystemExit("data YAML does not exist: %s" % args.data)
    if not os.path.isfile(args.model):
        raise SystemExit("starting model does not exist: %s" % args.model)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required: %s" % exc)

    model = YOLO(args.model, task="detect")
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project_dir,
        name=args.name,
        exist_ok=False,
        patience=max(15, min(30, args.epochs // 3)),
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        close_mosaic=min(10, max(0, args.epochs // 5)),
        hsv_h=0.02,
        hsv_s=0.50,
        hsv_v=0.40,
        degrees=3.0,
        translate=0.08,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        # Never horizontally flip directional traffic signs unless the label
        # itself is also swapped LEFT <-> RIGHT.
        fliplr=0.0,
        mosaic=0.60,
        mixup=0.10,
        plots=True,
        verbose=True,
    )
    print("train_save_dir=%s" % result.save_dir)
    print("best=%s" % os.path.join(str(result.save_dir), "weights", "best.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
