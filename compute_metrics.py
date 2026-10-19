import argparse
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO


def parse_label(txt_path, w, h):
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = [float(p) for p in parts[1:5]]
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h
            boxes.append((cls, (x1, y1, x2, y2)))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(prec, rec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.shape[0] - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def main():
    ap = argparse.ArgumentParser(description='YOLO GT-based metrics: P/R/F1, mAP@0.5, mAP@0.5:0.95, threshold sweep')
    ap.add_argument('--model', required=True, help='path to best.pt')
    ap.add_argument('--images', required=True, help='val images folder')
    ap.add_argument('--labels', required=True, help='val labels folder (YOLO txt format)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--name', default='results', help='output file prefix')
    ap.add_argument(
        '--conf-sweep',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    args = ap.parse_args()

    model = YOLO(args.model)
    names = model.names
    nclass = len(names)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = []
    for root, _, files in os.walk(args.images):
        for f in files:
            if f.lower().endswith(exts):
                image_files.append(os.path.join(root, f))
    image_files.sort()

    gt_by_cls = {}
    img_keys = []
    detections = []
    missing = 0
    for img_path in image_files:
        rel = os.path.relpath(img_path, args.images)
        stem = os.path.splitext(rel)[0]
        label_file = os.path.join(args.labels, stem + '.txt')
        if not os.path.exists(label_file):
            missing += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            print('cannot read:', img_path)
            continue
        h, w = img.shape[:2]
        idx = len(img_keys)
        img_keys.append(img_path)
        for cls, box in parse_label(label_file, w, h):
            gt_by_cls.setdefault(cls, {}).setdefault(idx, []).append(box)
        result = model.predict(source=img_path, imgsz=args.imgsz, conf=0.001, iou=0.5, device=args.device, verbose=False)[0]
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append((conf, cls, idx, (x1, y1, x2, y2)))

    print(f'images: total={len(image_files)} with_gt={len(img_keys)} missing_labels={missing}')
    print(f'detections (conf>=0.001): {len(detections)}')

    det_by_cls = {c: [] for c in range(nclass)}
    for d in detections:
        det_by_cls[d[1]].append(d)

    def curve_at_iou(iou_thr):
        out = {}
        for c in range(nclass):
            dets = sorted(det_by_cls[c], key=lambda d: d[0], reverse=True)
            used_c = {im: [False] * len(bx) for im, bx in gt_by_cls.get(c, {}).items()}
            tp = []
            for d in dets:
                conf, cls, im, box = d
                best = -1
                best_iou = iou_thr
                for i, gb in enumerate(gt_by_cls.get(c, {}).get(im, [])):
                    if used_c.get(im, [])[i]:
                        continue
                    v = iou(box, gb)
                    if v >= best_iou:
                        best_iou = v
                        best = i
                if best >= 0:
                    used_c[im][best] = True
                    tp.append(1)
                else:
                    tp.append(0)
            n_gt = sum(len(v) for v in gt_by_cls.get(c, {}).values())
            confs = [d[0] for d in dets]
            out[c] = (confs, tp, n_gt)
        return out

    def prec_rec_ap(confs, tp, n_gt):
        if not confs:
            return np.array([0.0]), np.array([0.0]), 0.0
        prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
        rec = np.cumsum(tp) / n_gt if n_gt > 0 else np.zeros(len(tp))
        return prec, rec, compute_ap(prec, rec)

    aps50 = {}
    stats50 = {}
    pr_curves = {}
    curve = curve_at_iou(0.5)
    for c in range(nclass):
        prec, rec, a50 = prec_rec_ap(*curve[c])
        aps50[c] = a50
        stats50[c] = (curve[c][2], int(sum(curve[c][1])))
        pr_curves[c] = (prec, rec)
    map50 = float(np.mean(list(aps50.values())))

    aps5095 = {c: [] for c in range(nclass)}
    for iou_thr in np.arange(0.55, 1.0, 0.05):
        crv = curve_at_iou(float(iou_thr))
        for c in range(nclass):
            aps5095[c].append(prec_rec_ap(*crv[c])[2])
    apr95 = {c: float(np.mean(v)) for c, v in aps5095.items()}
    map5095 = float(np.mean(list(apr95.values())))

    print('\n=== mAP ===')
    for c in range(nclass):
        print(f'{names[c]:<18} AP50={aps50[c]:.4f}  AP50:95={apr95[c]:.4f}  GT={stats50[c][0]}  matchedTP={stats50[c][1]}')
    print(f'{"ALL":<18} mAP@0.5={map50:.4f}  mAP@0.5:0.95={map5095:.4f}')

    rows = []
    best_f1 = 0.0
    best_t = args.conf_sweep[0]
    for t in args.conf_sweep:
        row = {'threshold': t}
        f1s = []
        for c in range(nclass):
            confs_all, tp_all, n_gt = curve[c]
            keep_idx = [i for i, cc in enumerate(confs_all) if cc >= t]
            tp = sum(tp_all[i] for i in keep_idx)
            fp = len(keep_idx) - tp
            fn = n_gt - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / n_gt if n_gt > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            row[f'P_{c}'] = round(p, 4)
            row[f'R_{c}'] = round(r, 4)
            row[f'F1_{c}'] = round(f1, 4)
            f1s.append(f1)
        row['F1_all'] = round(float(np.mean(f1s)), 4)
        rows.append(row)
        if row['F1_all'] > best_f1:
            best_f1 = row['F1_all']
            best_t = t

    print('\n=== Threshold sweep @IoU0.5 ===')
    print('conf  | ' + ' | '.join(f'P_{i}' for i in range(nclass)) + ' | ' + ' | '.join(f'R_{i}' for i in range(nclass)) + ' | F1_all')
    for row in rows:
        vals = [f'{row["threshold"]:.2f}']
        vals += [f'{row[f"P_{i}"]:.3f}' for i in range(nclass)]
        vals += [f'{row[f"R_{i}"]:.3f}' for i in range(nclass)]
        vals.append(f'{row["F1_all"]:.3f}')
        print(' | '.join(vals))
    print(f'\nBEST F1_all = {best_f1:.4f}  at threshold = {best_t:.2f}')

    header = ['threshold'] + [f'P_{c}' for c in range(nclass)] + [f'R_{c}' for c in range(nclass)] + [f'F1_{c}' for c in range(nclass)] + ['F1_all']
    with open(args.name + '_threshold_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    with open(args.name + '_map.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'AP50', 'AP50:95', 'GT_count', 'matched_TP'])
        for c in range(nclass):
            w.writerow([names[c], round(aps50[c], 4), round(apr95[c], 4), stats50[c][0], stats50[c][1]])
        w.writerow(['ALL', round(map50, 4), round(map5095, 4), '', ''])

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for c in range(nclass):
            prec, rec = pr_curves[c]
            plt.figure()
            plt.plot(rec, prec, 'b-')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR curve IoU@0.5: {names[c]} (AP50={aps50[c]:.3f})')
            plt.xlim(0, 1)
            plt.ylim(0, 1.05)
            plt.grid(True)
            plt.savefig(f'{args.name}_pr_{names[c]}.png', dpi=150)
            plt.close()

        plt.figure()
        plt.plot([r['threshold'] for r in rows], [r['F1_all'] for r in rows], 'r-o')
        plt.xlabel('Confidence threshold')
        plt.ylabel('F1 (all classes)')
        plt.title('F1 vs confidence threshold')
        plt.grid(True)
        plt.savefig(f'{args.name}_f1_sweep.png', dpi=150)
        plt.close()
        print('plotlar saqlandi: *_pr_*.png, *_f1_sweep.png')
    except Exception as e:
        print('plotlar yaratilmadi (matplotlib xatosi):', e)

    print('fayllar saqlandi:', args.name + '_map.csv', args.name + '_threshold_sweep.csv')


if __name__ == '__main__':
    main()