#!/usr/bin/env python3
"""
evaluate_detector.py  — Member C, Task 3
Evaluate YOLOv8n precision/recall across multiple confidence thresholds
on a folder of aerial images.

Usage:
  python3 scripts/evaluate_detector.py \
      --images sar_world/dataset/frames \
      --labels sar_world/dataset/annot \
      --model   yolov8n.pt \
      --output  results/detection_eval.png

Labels format (YOLO .txt per image, same stem):
  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>  (class 0 = person)

Output:
  - Precision-Recall curve image (results/pr_curve.png)
  - Table printed to stdout  (also saved as results/detection_pr_table.csv)
  - results/detection_eval.png  (bar chart of metrics at best F1 threshold)
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── IoU helper ─────────────────────────────────────────────────────────────────
def box_iou(boxA, boxB) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter)


# ── Load ground truth ──────────────────────────────────────────────────────────
def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> List[List[float]]:
    """Load YOLO-format label file → list of [x1,y1,x2,y2] pixel boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls != 0:
                continue  # person only
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


# ── Per-threshold evaluation ────────────────────────────────────────────────────
def evaluate_at_conf(
    model,
    image_paths: List[Path],
    label_dir: Path,
    conf: float,
    iou_thresh: float = 0.50,
    imgsz: int = 640,
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """
    Run YOLO at given conf threshold over all images.
    Returns (precision, recall, f1).
    """
    import cv2

    tp = fp = fn = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, w, h)

        # Predictions
        results = model.predict(
            img,
            classes=[0],
            conf=conf,
            iou=0.5,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        pred_boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append([x1, y1, x2, y2])

        # Match predictions to ground truth (greedy, IoU >= iou_thresh)
        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",  required=True,  help="Directory of JPEG/PNG images")
    ap.add_argument("--labels",  required=True,  help="Directory of YOLO .txt label files")
    ap.add_argument("--model",   default="yolov8n.pt")
    ap.add_argument("--device",  default="cpu")
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--output",  default="results")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    img_dir   = Path(args.images)
    label_dir = Path(args.labels)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    )
    if not image_paths:
        print(f"No images found in {img_dir}")
        sys.exit(1)
    print(f"Evaluating on {len(image_paths)} images...")

    model = YOLO(args.model)

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    rows = []
    print(f"\n{'conf':>6}  {'precision':>10}  {'recall':>8}  {'F1':>8}")
    print("-" * 40)
    for conf in thresholds:
        p, r, f1 = evaluate_at_conf(
            model, image_paths, label_dir, conf,
            imgsz=args.imgsz, device=args.device
        )
        rows.append({"conf": conf, "precision": p, "recall": r, "f1": f1})
        print(f"{conf:>6.2f}  {p:>10.4f}  {r:>8.4f}  {f1:>8.4f}")

    # Save CSV
    csv_path = out_dir / "detection_pr_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conf", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTable saved: {csv_path}")

    # Best F1 threshold
    best = max(rows, key=lambda r: r["f1"])
    print(f"\nBest F1={best['f1']:.4f} at conf={best['conf']:.2f}  "
          f"(P={best['precision']:.4f}, R={best['recall']:.4f})")

    # ── P-R curve ─────────────────────────────────────────────────────────────
    confs = [r["conf"]      for r in rows]
    precs = [r["precision"] for r in rows]
    recs  = [r["recall"]    for r in rows]
    f1s   = [r["f1"]        for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e2e2e2")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    # PR curve
    axes[0].plot(recs, precs, color="#e94560", linewidth=2, marker="o", markersize=5)
    axes[0].set_xlabel("Recall",    color="#e2e2e2")
    axes[0].set_ylabel("Precision", color="#e2e2e2")
    axes[0].set_title("Precision-Recall Curve", color="#e2e2e2")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].grid(color="#333366", linewidth=0.5, alpha=0.5)

    # F1 vs conf
    axes[1].plot(confs, f1s,   color="#0f3460", linewidth=2, label="F1")
    axes[1].plot(confs, precs, color="#00b4d8", linewidth=1.5, linestyle="--", label="Precision")
    axes[1].plot(confs, recs,  color="#f4a261", linewidth=1.5, linestyle="--", label="Recall")
    axes[1].axvline(best["conf"], color="#ffd166", linewidth=1.5,
                    linestyle=":", label=f"Best F1 @ {best['conf']:.2f}")
    axes[1].set_xlabel("Confidence threshold", color="#e2e2e2")
    axes[1].set_ylabel("Score", color="#e2e2e2")
    axes[1].set_title("Metrics vs Confidence Threshold", color="#e2e2e2")
    axes[1].set_xlim(min(confs), max(confs)); axes[1].set_ylim(0, 1)
    axes[1].legend(facecolor="#0d1117", labelcolor="#e2e2e2", fontsize=9)
    axes[1].grid(color="#333366", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    eval_path = out_dir / "detection_eval.png"
    plt.savefig(eval_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {eval_path}")


if __name__ == "__main__":
    main()
