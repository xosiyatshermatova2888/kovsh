"""VAZIFA 2: Real vaqtli FPS / inference-tezlik benchmark (GPU va CPU).

Model: eng yangi 7 klassli best.pt
Video: static/video.mp4
Nasl: GPU (device=0) va CPU (device='cpu') uchun alohida olchab,
ikkalasini taqqoslash jadvali chiqariladi.
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def load_frames(source: str) -> list:
    cap = cv2.VideoCapture(source)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def benchmark(model: YOLO, frames: list, device: str, imgsz: int,
              warmup: int) -> dict:
    for i in range(min(warmup, len(frames))):
        model.predict(source=frames[i], imgsz=imgsz, device=device,
                      conf=0.25, verbose=False)

    times = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict(source=frame, imgsz=imgsz, device=device,
                      conf=0.25, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    return {
        "device": str(device),
        "frames": len(t),
        "avg_ms": float(t.mean()),
        "median_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
        "fps": 1000.0 / float(t.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="GPU/CPU FPS benchmark")
    ap.add_argument("--model", default="runs/kovsh_motion_exp1/weights/best.pt")
    ap.add_argument("--source", default="static/video.mp4")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    model = YOLO(args.model)
    nc = len(model.names)
    print("Model : {}  ({} klass)".format(args.model, nc))
    print("Video : {}".format(args.source))
    print("imgsz : {}\n".format(args.imgsz))

    frames = load_frames(args.source)
    print("Kadrlar soni: {}\n".format(len(frames)))

    results = [benchmark(model, frames, "0", args.imgsz, args.warmup),
               benchmark(model, frames, "cpu", args.imgsz, args.warmup)]

    print("=" * 74)
    print("{:<10}{:<8}{:<10}{:<11}{:<9}{:<8}{:<8}{:<8}".format(
        "Qurilma", "Kadr", "avg_ms", "median_ms", "p95_ms", "min", "max",
        "FPS"))
    print("-" * 74)
    for r in results:
        dname = "GPU" if r["device"] == "0" else "CPU"
        print("{:<10}{:<8}{:<10.1f}{:<11.1f}{:<9.1f}{:<8.1f}{:<8.1f}"
              "{:<8.1f}".format(dname, r["frames"], r["avg_ms"],
                                r["median_ms"], r["p95_ms"], r["min_ms"],
                                r["max_ms"], r["fps"]))
    print("=" * 74)

    gpu = results[0]
    cpu = results[1]
    msspeed = (cpu["avg_ms"] / gpu["avg_ms"] - 1) * 100
    fspeed = (cpu["fps"] / gpu["fps"] - 1) * 100
    print("\nXulosa: GPU ortacha {:.1f} FPS ({:.1f} ms), CPU ortacha "
          "{:.1f} FPS ({:.1f} ms).".format(gpu["fps"], gpu["avg_ms"],
                                           cpu["fps"], cpu["avg_ms"]))
    print("GPU CPU dan ~{:.0f}% tez (inference vaqti), "
          "FPS bo'yicha ~{:.0f}% farq.".format(msspeed, fspeed))


if __name__ == "__main__":
    main()