import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='FPS / inference time benchmark for YOLO')
    ap.add_argument('--model', required=True, help='path to best.pt')
    ap.add_argument('--source', required=True, help='video.mp4 or images folder')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--max-frames', type=int, default=100)
    args = ap.parse_args()

    model = YOLO(args.model)

    frames = []
    if os.path.isdir(args.source):
        for f in sorted(os.listdir(args.source)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                p = os.path.join(args.source, f)
                im = cv2.imread(p)
                if im is not None:
                    frames.append(im)
    else:
        cap = cv2.VideoCapture(args.source)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()

    if not frames:
        print('kadrlar topilmadi')
        return

    print(f'kadrlar: {len(frames)}, imgsz: {args.imgsz}, device: {args.device}')

    for i in range(min(args.warmup, len(frames))):
        model.predict(source=frames[i], imgsz=args.imgsz, device=args.device, conf=args.conf, verbose=False)

    times = []
    for i in range(min(args.max_frames, len(frames))):
        t0 = time.perf_counter()
        model.predict(source=frames[i], imgsz=args.imgsz, device=args.device, conf=args.conf, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    print(f'o\'rtacha inference: {times.mean():.1f} ms/kadr')
    print(f'FPS (o\'rtacha): {1000.0 / times.mean():.1f}')
    print(f'median: {np.percentile(times, 50):.1f} ms,  p95: {np.percentile(times, 95):.1f} ms,  min: {times.min():.1f} ms,  max: {times.max():.1f} ms')


if __name__ == '__main__':
    main()