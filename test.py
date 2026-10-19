"""Test datasetda best.pt bilan prediction.

Natijalar:
    results/test/images/      (bounding box chizilgan rasmlar)
    results/test/detections.csv
    results/test/test_summary.csv

Ishga tushirish:
    python test.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ensure_dirs


def main() -> int:
    from ultralytics import YOLO

    ensure_dirs()
    best = Path(__file__).resolve().parent / "results" / "training" / "best.pt"
    if not best.exists():
        best = Path(__file__).resolve().parent / "runs" / "teeth_truck_exp1" / "weights" / "best.pt"
    test_img_dir = Path(__file__).resolve().parent / "data" / "test" / "images"
    if not best.exists():
        print("[XATO] best.pt topilmadi.", file=sys.stderr)
        return 1
    if not test_img_dir.exists():
        print("[XATO] test papkasi topilmadi (data/test/images).", file=sys.stderr)
        return 1

    import config

    out_dir = config.RESULTS_DIR / "test"
    out_img = out_dir / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    for f in out_img.glob("*"):
        f.unlink()

    print("=" * 62)
    print("TEST PREDICTION (best.pt)")
    print("=" * 62)
    model = YOLO(str(best))
    names = model.names

    imgs = sorted([p for p in test_img_dir.glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])

    rows = []
    class_count = Counter()
    total_conf = 0.0
    n_det = 0
    images_with_det = 0

    for p in imgs:
        r = model.predict(str(p), conf=config.INFERENCE_CONF, iou=config.INFERENCE_IOU,
                          imgsz=config.IMG_SIZE, device=config.DEVICE, verbose=False)[0]
        annotated = r.plot(font_size=1, line_width=2, labels=True, conf=True)
        out_name = out_img / f"{p.stem}.jpg"
        cv2.imwrite(str(out_name), annotated)
        det = len(r.boxes) if r.boxes is not None else 0
        if det > 0:
            images_with_det += 1
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                rows.append({
                    "image": p.name, "class_id": cls, "class": names[cls],
                    "confidence": round(conf, 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                })
                class_count[names[cls]] += 1
                total_conf += conf
                n_det += 1

    with open(out_dir / "detections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else
                                ["image", "class_id", "class", "confidence",
                                 "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "images": len(imgs),
        "images_with_detections": images_with_det,
        "total_detections": n_det,
        "mean_confidence": round(total_conf / n_det, 4) if n_det else None,
    }
    for name, cnt in class_count.items():
        summary[f"detections_{name}"] = cnt

    with open(out_dir / "test_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for k, v in summary.items():
            writer.writerow([k, v])

    print(f"rasmlar           : {len(imgs)}")
    print(f"deteksiya bor     : {images_with_det}")
    print(f"jami deteksiya    : {n_det}")
    print(f"class tagsimoti   : {dict(class_count)}")
    print(f"annotated rasmlar : {out_img}")
    print(f"detections.csv    : {out_dir / 'detections.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())