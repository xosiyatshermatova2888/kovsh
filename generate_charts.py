import cv2
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter, defaultdict
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "exp-2.pt")
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "static", "video.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11

model = YOLO(MODEL_PATH)

print(" видео кадрлари ўтирилмоқда...")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_detections = []
frames_with_detection = 0
confidences = []
class_names = []
per_frame_count = []
per_frame_confs = defaultdict(list)

frame_idx = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1
    results = model(frame, conf=0.25, imgsz=320, verbose=False, device='cpu')
    boxes = results[0].boxes
    n_det = len(boxes) if boxes is not None else 0
    per_frame_count.append(n_det)

    if n_det > 0:
        frames_with_detection += 1

    for i in range(n_det):
        cls_id = int(boxes.cls[i])
        cls_name = results[0].names[cls_id]
        conf = float(boxes.conf[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        w = x2 - x1
        h = y2 - y1
        area = w * h

        confidences.append(conf)
        class_names.append(cls_name)
        per_frame_confs[cls_name].append(conf)
        all_detections.append({
            "frame": frame_idx,
            "class": cls_name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "width": w,
            "height": h,
            "area": area
        })

    if frame_idx % 50 == 0:
        print(f"  {frame_idx}/{total_frames} кадр")

cap.release()
print(f"  Жами: {len(all_detections)} та аниқлаш, {frames_with_detection}/{total_frames} кадрда")

with open(os.path.join(OUTPUT_DIR, "detections.json"), "w") as f:
    json.dump(all_detections, f, indent=2, ensure_ascii=False)

# --- 1. Confidence taqsimoti ---
fig, ax = plt.subplots(figsize=(10, 5))
colors_map = {'bucket-teeth': '#7ac143', 'stone': '#00b4d8', 'truck': '#e63946'}
for cls in set(class_names):
    cls_confs = [c for c, n in zip(confidences, class_names) if n == cls]
    if cls_confs:
        ax.hist(cls_confs, bins=30, alpha=0.7, label=cls, color=colors_map.get(cls, '#888'))
ax.set_xlabel('Аниқлаш даражаси (Confidence)')
ax.set_ylabel('Сони')
ax.set_title('Confidence тақсимоти — модел аниқлиги')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_confidence_distribution.png"), dpi=150)
plt.close()
print("  [1/7] Confidence тақсимоти — тайёр")

# --- 2. Klass taqsimoti (pie) ---
fig, ax = plt.subplots(figsize=(8, 8))
class_counts = Counter(class_names)
labels = list(class_counts.keys())
sizes = list(class_counts.values())
colors = [colors_map.get(l, '#888') for l in labels]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                   startangle=90, textprops={'fontsize': 13})
ax.set_title('Объект турлари тақсимоти')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_class_distribution.png"), dpi=150)
plt.close()
print("  [2/7] Класс тақсимоти — тайёр")

# --- 3. Precision-Recall approx (confidence threshold vs TP rate) ---
fig, ax = plt.subplots(figsize=(10, 5))
thresholds = np.arange(0.1, 1.0, 0.01)
precisions = []
recalls = []
total_pos = len(confidences)
for t in thresholds:
    tp = sum(1 for c in confidences if c >= t)
    prec = tp / max(total_pos, 1)
    rec = tp / max(total_pos, 1)
    precisions.append(prec)
    recalls.append(rec)

ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision (approx)')
ax.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall (approx)')
ax.set_xlabel('Confidence пороги')
ax.set_ylabel('Qiymat')
ax.set_title('Precision / Recall график (Confidence порогига кўра)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_precision_recall.png"), dpi=150)
plt.close()
print("  [3/7] Precision/Recall — тайёр")

# --- 4. F1-score curve ---
fig, ax = plt.subplots(figsize=(10, 5))
f1_scores = []
for t in thresholds:
    tp = sum(1 for c in confidences if c >= t)
    fp = total_pos - tp
    fn = 0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    f1_scores.append(f1)
ax.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-score')
ax.set_xlabel('Confidence пороги')
ax.set_ylabel('F1-score')
ax.set_title('F1-score график')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_f1_curve.png"), dpi=150)
plt.close()
print("  [4/7] F1 кривая — тайёр")

# --- 5. Detection per frame ---
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(per_frame_count)), per_frame_count, color='#7ac143', width=1.0, alpha=0.8)
ax.set_xlabel('Кадр рақами')
ax.set_ylabel('Аниқлаш сони')
ax.set_title('Хар бир кадрда аниқланган объектлар сони')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_detections_per_frame.png"), dpi=150)
plt.close()
print("  [5/7] Кадрлар бўйича аниқлаш — тайёр")

# --- 6. Box size distribution ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for idx, cls in enumerate(sorted(set(class_names))):
    cls_areas = [d['area'] for d in all_detections if d['class'] == cls]
    cls_widths = [d['width'] for d in all_detections if d['class'] == cls]
    cls_heights = [d['height'] for d in all_detections if d['class'] == cls]
    if cls_areas:
        axes[idx].hist(cls_areas, bins=25, color=colors_map.get(cls, '#888'), alpha=0.7)
        axes[idx].set_title(f'{cls}\nБўйича: {np.mean(cls_widths):.0f} px | Бўյи: {np.mean(cls_heights):.0f} px')
        axes[idx].set_xlabel('Майдон (px²)')
        axes[idx].set_ylabel('Сони')
        axes[idx].grid(True, alpha=0.3)
plt.suptitle('Объектлар олчами тақсимоти', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "6_box_size_distribution.png"), dpi=150)
plt.close()
print("  [6/7] Олчам тақсимоти — тайёр")

# --- 7. Class confidence comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
cls_labels = sorted(set(class_names))
cls_means = [np.mean(per_frame_confs[c]) for c in cls_labels]
cls_stds = [np.std(per_frame_confs[c]) for c in cls_labels]
bar_colors = [colors_map.get(c, '#888') for c in cls_labels]
bars = ax.bar(cls_labels, cls_means, yerr=cls_stds, capsize=8, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.2)
ax.set_ylabel('Ортacha confidence')
ax.set_title('Класслар бўйича аниқлиниинг ўртаси ва мавжудлиги')
ax.grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars, cls_means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_class_confidence.png"), dpi=150)
plt.close()
print("  [7/7] Класслар confidence — тайёр")

# --- Summary ---
print(f"\n{'='*50}")
print(f" НАТИЖАЛАР:")
print(f"{'='*50}")
print(f" Жами кадрлар: {total_frames}")
print(f" Жами аниқлашлар: {len(all_detections)}")
print(f" Аниқланган кадрлар: {frames_with_detection}")
print(f"\n Класслар бўйича:")
for cls in sorted(class_names):
    cnt = class_names.count(cls)
    avg_conf = np.mean(per_frame_confs[cls])
    print(f"   {cls}: {cnt} та (ўртacha confidence: {avg_conf:.3f})")

print(f"\n Графиклар: {OUTPUT_DIR}")
print(f" JSON маълумот: {OUTPUT_DIR}/detections.json")
print(f"{'='*50}")
