import os
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

BASE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE, "output", "detections.json")
OUT_DIR = os.path.join(BASE, "output", "revised")
os.makedirs(OUT_DIR, exist_ok=True)

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13

COLORS = {'bucket-teeth': '#7ac143', 'truck': '#e63946'}

data = json.load(open(JSON_PATH, encoding='utf-8'))
classes = {}
for x in data:
    classes.setdefault(x['class'], []).append(x)

total = len(data)
cls_order = ['bucket-teeth', 'truck']
cls_order = [c for c in cls_order if c in classes] + [c for c in classes if c not in cls_order]

summary = "REAL NATIJALAR JAMLANMASI\n"
summary += "=" * 60 + "\n"
summary += f"Video: static/video.mp4  |  Kadrlar: 200  |  Deteksiya: {total}\n"
summary += f"Model: exp-2.pt  |  conf>=0.25  |  imgsz=320  |  resolution 2592x1944\n"
summary += "=" * 60 + "\n"
for cls in cls_order:
    items = classes[cls]
    confs = np.array([x['confidence'] for x in items])
    w = np.array([x['width'] for x in items])
    h = np.array([x['height'] for x in items])
    a = np.array([x['area'] for x in items])
    summary += f"\n{cls}:\n"
    summary += f"  Soni: {len(items)}  ({100*len(items)/total:.1f}%)\n"
    summary += f"  Confidence: ort={confs.mean():.3f}  std={confs.std():.3f}  min={confs.min():.3f}  max={confs.max():.3f}\n"
    summary += f"  Bbox: kenglik_ort={w.mean():.0f}  balandlik_ort={h.mean():.0f}\n"
    summary += f"  Maydon: ort={a.mean():.0f}  med={np.median(a):.0f}  [min={a.min():.0f}, max={a.max():.0f}]\n"

summary += "\n" + "=" * 60 + "\nRETENTION TAXLILI (asl Precision/Recall emas!):\n"
for cls in cls_order:
    confs = np.array([x['confidence'] for x in classes[cls]])
    summary += f"\n{cls} (n={len(confs)}):\n"
    for t in [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
        n = int((confs >= t).sum())
        summary += f"  >= {t:.2f}: {n}  ({100*n/len(confs):.0f}%)\n"

summary += "\nFPS benchmark (CPU):\n  imgsz=640 -> 64.3 ms/kadr, 15.6 FPS\n  Video tezligi: 20 fps\n  GPU: o'lchash kutilmoqda\n"

with open(os.path.join(OUT_DIR, "natijalar_jamlanmasi.txt"), "w", encoding="utf-8") as f:
    f.write(summary)
print(summary)

# ---------- 1. Confidence distribution ----------
fig, ax = plt.subplots(figsize=(10, 5))
for cls in cls_order:
    confs = [x['confidence'] for x in classes[cls]]
    ax.hist(confs, bins=30, alpha=0.7, label=cls, color=COLORS[cls])
ax.axvline(0.25, color='k', ls='--', lw=1.2)
ax.text(0.27, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 1, 'chegara 0.25', fontsize=9)
ax.set_xlabel('Confidence')
ax.set_ylabel('Deteksiyalar soni')
ax.set_title('Har bir sinf bo\'yicha confidence taqsimoti')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "1_confidence_distribution.png"), dpi=150)
plt.close()

# ---------- 2. Class distribution (pie) ----------
fig, ax = plt.subplots(figsize=(8, 8))
counts = Counter(x['class'] for x in data)
labels_p = [c for c in cls_order]
sizes = [counts[c] for c in labels_p]
colors = [COLORS[c] for c in labels_p]
wedges, texts, autotexts = ax.pie(sizes, labels=labels_p, colors=colors, autopct='%1.1f%%',
                                  startangle=90, textprops={'fontsize': 13})
ax.set_title('Deteksiyalarning sinflar bo\'yicha taqsimoti')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "2_class_distribution.png"), dpi=150)
plt.close()

# ---------- 3. Retention rate vs threshold (HONEST) ----------
fig, ax = plt.subplots(figsize=(10, 5))
thresholds = np.arange(0.25, 0.96, 0.01)
for cls in cls_order:
    confs = np.array([x['confidence'] for x in classes[cls]])
    rates = [100 * float((confs >= t).sum()) / len(confs) for t in thresholds]
    ax.plot(thresholds, rates, linewidth=2, label=cls, color=COLORS[cls])
ax.axvspan(0.25, 0.30, color='gray', alpha=0.15, label='Taklif qilingan chegara')
ax.set_xlabel('Confidence chegarasi')
ax.set_ylabel('Saqlanib qolgan deteksiyalar ulushi (%)')
ax.set_title('Retention egri: confidence chegarasiga qarab deteksiyalarning saqlanishi')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0.25, 0.95)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "3_retention_curve.png"), dpi=150)
plt.close()

# ---------- 4. Retention-based F1 (HONEST label) ----------
fig, ax = plt.subplots(figsize=(10, 5))
for cls in cls_order:
    confs = np.array([x['confidence'] for x in classes[cls]])
    f1s = []
    for t in thresholds:
        keep = float((confs >= t).sum())
        prec = keep / total if total else 0.0
        rec = keep / len(confs) if len(confs) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    ax.plot(thresholds, f1s, linewidth=2, label=cls, color=COLORS[cls])
ax.axvspan(0.25, 0.30, color='gray', alpha=0.15, label='Taklif qilingan chegara')
ax.set_xlabel('Confidence chegarasi')
ax.set_ylabel('F1 (retention asosida)')
ax.set_title('Retention asosidagi F1 (haqiqiy Ground-Truth F1 emas)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0.25, 0.95)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "4_f1_retention.png"), dpi=150)
plt.close()

# ---------- 5. Detections per frame ----------
fig, ax = plt.subplots(figsize=(12, 4))
frames = [0] * 200
for x in data:
    frames[x['frame'] - 1] += 1
ax.bar(range(1, 201), frames, color='#7ac143', width=1.0, alpha=0.8)
ax.set_xlabel('Kadr raqami')
ax.set_ylabel('Deteksiyalar soni')
ax.set_title('Har bir kadrda aniqlangan obyektlar soni')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(0, 200)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "5_detections_per_frame.png"), dpi=150)
plt.close()

# ---------- 6. Box size distribution ----------
fig, axes = plt.subplots(1, len(cls_order), figsize=(6.5 * len(cls_order), 4.5))
if len(cls_order) == 1:
    axes = [axes]
for ax, cls in zip(axes, cls_order):
    areas = [x['area'] for x in classes[cls]]
    w = np.mean([x['width'] for x in classes[cls]])
    h = np.mean([x['height'] for x in classes[cls]])
    ax.hist(areas, bins=25, color=COLORS[cls], alpha=0.7)
    ax.set_title(f'{cls}\nort. bbox: {w:.0f} x {h:.0f} px')
    ax.set_xlabel('Maydon (px²)')
    ax.set_ylabel('Soni')
    ax.grid(True, alpha=0.3)
    if cls == 'truck':
        ax.set_xscale('log')
fig.suptitle('Obyekt o\'lchami taqsimoti', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "6_box_size_distribution.png"), dpi=150)
plt.close()

# ---------- 7. Class confidence comparison ----------
fig, ax = plt.subplots(figsize=(9, 5))
means = [np.mean([x['confidence'] for x in classes[c]]) for c in cls_order]
stds = [np.std([x['confidence'] for x in classes[c]]) for c in cls_order]
colors = [COLORS[c] for c in cls_order]
bars = ax.bar(cls_order, means, yerr=stds, capsize=8, color=colors, alpha=0.85, edgecolor='white', lw=1.2)
ax.set_ylabel('O\'rtacha confidence')
ax.set_title('Sinflar bo\'yicha o\'rtacha confidence va standart chetlanish')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f'{m:.3f}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "7_class_confidence.png"), dpi=150)
plt.close()

print("\nGRAFIKLAR:", OUT_DIR)