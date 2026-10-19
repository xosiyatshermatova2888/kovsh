"""Validation: best.pt ni validation/test datasetda baholaydi.

Metrikalar (Overall va har bir class bo'yicha):
    Precision, Recall, F1-score, mAP50, mAP50-95

Natijalar:
    results/validation/metrics.csv
    results/validation/metrics.xlsx
    results/validation/val/   (Ultralytics confusion matrix, PR/P/R/F1 curves)

Ishga tushirish:
    python validate.py            # val split
    python validate.py test       # test split
"""

import sys

import csv
import math
import shutil
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import ensure_dirs


def safe_f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def val_at(arr, i: int, present: list, pos: dict) -> float:
    """Class ko'rrsatkichini qaytaradi; klass umuman bo'lmasa (nan) 0.0."""
    if len(arr) == len(present):
        # Ultralytics faqat mavjud classlarni qaytarganda index ni pos orqali
        # to'g'ri class'ga bog'laymiz.
        v = float(arr[pos[i]]) if i in pos else float("nan")
    else:
        v = float(arr[i]) if i < len(arr) else float("nan")
    return 0.0 if math.isnan(v) else v


def metrics_to_csv(metrics_rows: list, csv_path: Path) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)


def main() -> int:
    from ultralytics import YOLO

    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    if split not in ("val", "test"):
        print(f"[XATO] nomalum split: {split} (val|test)", file=sys.stderr)
        return 1

    ensure_dirs()
    best = config.RESULTS_DIR / "training" / "best.pt"
    if not best.exists():
        best = config.EXP_DIR / "weights" / "best.pt"
    if not best.exists():
        print(f"[XATO] best.pt topilmadi ({config.EXP_DIR / 'weights' / 'best.pt'}). "
              f"Avval `train.py` ishga tushiring.", file=sys.stderr)
        return 1

    print("=" * 62)
    print(f"VALIDATION ({split}, best.pt)")
    print("=" * 62)
    model = YOLO(str(best))
    val_dir = config.RESULTS_DIR / "validation" / split
    if val_dir.exists():
        shutil.rmtree(val_dir)
    val_dir.mkdir(parents=True)

    metrics = model.val(
        data=str(config.DATA_YAML),
        split=split,
        imgsz=config.IMG_SIZE,
        device=config.DEVICE,
        task="detect",
        project=str(config.RESULTS_DIR / "validation"),
        name=split,
        exist_ok=True,
        plots=True,
    )

    names = model.names
    order = sorted(names.keys())

    # Split ichida haqiqiy mavjud classlar (kovsh_motion/tuck kabi bo'sh
    # classlar Ultralytics natija massivlaridan chiqarib tashlanadi).
    lbl_dir = config.DATA_DIR / split / "labels"
    present = set()
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                present.add(int(line.split()[0]))
    present = sorted(present)
    pos = {cid: idx for idx, cid in enumerate(present)}

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    f1_all = safe_f1(mp, mr)

    p_all = dict(zip(order, [val_at(metrics.box.p, i, present, pos) for i in order]))
    r_all = dict(zip(order, [val_at(metrics.box.r, i, present, pos) for i in order]))
    ap50_all = dict(zip(order, [val_at(metrics.box.ap50, i, present, pos) for i in order]))
    ap_all = dict(zip(order, [val_at(metrics.box.ap, i, present, pos) for i in order]))

    overall_row = {
        "metric": "Overall",
        "precision": round(mp, 4), "recall": round(mr, 4),
        "f1": round(f1_all, 4),
        "map50": round(map50, 4), "map50_95": round(map5095, 4),
    }
    rows = [overall_row]
    for i in order:
        rows.append({
            "metric": names[i],
            "precision": round(p_all[i], 4), "recall": round(r_all[i], 4),
            "f1": round(safe_f1(p_all[i], r_all[i]), 4),
            "map50": round(ap50_all[i], 4), "map50_95": round(ap_all[i], 4),
        })

    df = pd.DataFrame(rows)
    metrics_csv = config.RESULTS_DIR / "validation" / f"metrics_{split}.csv"
    metrics_xlsx = config.RESULTS_DIR / "validation" / f"metrics_{split}.xlsx"
    metrics_to_csv(rows, metrics_csv)
    df.to_excel(metrics_xlsx, index=False)

    print()
    print(f"{'Metric':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}{'mAP50':>9}{'mAP50-95':>10}")
    for r in rows:
        print(f"{r['metric']:<14}{r['precision']:>10.4f}{r['recall']:>10.4f}"
              f"{r['f1']:>10.4f}{r['map50']:>9.4f}{r['map50_95']:>10.4f}")
    print()
    print(f"Ultralytics {split} fayllari: {config.RESULTS_DIR / 'validation' / split}")
    print(f"metrics.csv  : {metrics_csv}")
    print(f"metrics.xlsx : {metrics_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())