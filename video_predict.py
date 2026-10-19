"""Videoda best.pt bilan real-time inference (deteksiya + FPS).

Natijalar:
    results/video_result.mp4   (bounding box chizilgan video)
    results/video_detections.csv
    results/video_fps.csv      (har kadr / o'rtacha inference FPS)
    results/video_fps.txt

Ishga tushirish:
    python video_predict.py
"""

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ensure_dirs


def main() -> int:
    import config
    from ultralytics import YOLO

    ensure_dirs()
    best = config.RESULTS_DIR / "training" / "best.pt"
    if not best.exists():
        best = config.EXP_DIR / "weights" / "best.pt"
    video = config.SOURCE_VIDEO
    if not best.exists():
        print("[XATO] best.pt topilmadi.", file=sys.stderr)
        return 1
    if not video.exists():
        print(f"[XATO] video topilmadi: {video}", file=sys.stderr)
        return 1

    print("=" * 62)
    print("VIDEO INFERENCE (best.pt)")
    print("=" * 62)
    model = YOLO(str(best))
    names = model.names

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("[XATO] videoni ochib bo'lmadi.", file=sys.stderr)
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = config.RESULTS_DIR / "video_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps_video, (w, h))

    rows = []
    fps_rows = []
    inf_times = []
    class_count = Counter()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t0 = time.perf_counter()
        r = model.predict(frame, conf=config.INFERENCE_CONF, iou=config.INFERENCE_IOU,
                          imgsz=config.IMG_SIZE, device=config.DEVICE, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        inf_times.append(dt)
        fps_rows.append({"frame": idx, "inference_ms": round(dt, 2),
                         "fps": round(1000.0 / dt, 2) if dt > 0 else None})
        annotated = r.plot(font_size=1, line_width=2, labels=True, conf=True)
        writer.write(annotated)
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                rows.append({
                    "frame": idx, "class_id": cls, "class": names[cls],
                    "confidence": round(conf, 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                })
                class_count[names[cls]] += 1

    cap.release()
    writer.release()

    avg_ms = sum(inf_times) / len(inf_times) if inf_times else None
    avg_fps = 1000.0 / avg_ms if avg_ms else None
    p95_ms = sorted(inf_times)[int(len(inf_times) * 0.95) - 1] if inf_times else None

    with open(config.RESULTS_DIR / "video_detections.csv", "w", newline="") as f:
        writer2 = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else
                                 ["frame", "class_id", "class", "confidence",
                                  "x1", "y1", "x2", "y2"])
        writer2.writeheader()
        writer2.writerows(rows)

    with open(config.RESULTS_DIR / "video_fps.csv", "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=["frame", "inference_ms", "fps"])
        w2.writeheader()
        w2.writerows(fps_rows)
        w2.writerow({"frame": "AVG", "inference_ms": round(avg_ms, 2) if avg_ms else "N/A",
                     "fps": round(avg_fps, 2) if avg_fps else "N/A"})

    txt = (f"video: {video}\n"
           f"frames_processed: {idx}\n"
           f"avg_inference_ms: {round(avg_ms, 2) if avg_ms else 'N/A'}\n"
           f"avg_fps: {round(avg_fps, 2) if avg_fps else 'N/A'}\n"
           f"p95_inference_ms: {round(p95_ms, 2) if p95_ms else 'N/A'}\n"
           f"class_distribution: {dict(class_count)}\n")
    (config.RESULTS_DIR / "video_fps.txt").write_text(txt, encoding="utf-8")

    print(f"kadrlar          : {idx}")
    print(f"o'rtacha inference: {avg_ms:.1f} ms/kadr" if avg_ms else "N/A")
    print(f"o'rtacha FPS     : {avg_fps:.1f}" if avg_fps else "FPS: N/A")
    print(f"class tagsimoti  : {dict(class_count)}")
    print(f"video            : {out_video}")
    print(f"video_detections.csv : {config.RESULTS_DIR / 'video_detections.csv'}")
    print(f"video_fps.csv/.txt   : {config.RESULTS_DIR / 'video_fps.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())