"""Training grafiklari va confusion matrix/curves generatsiyasi.

Manbalar:
    - runs/teeth_truck_exp1/results.csv        (epoch bo'yicha losslar va metrikalar)
    - results/validation/val/                  (Ultralytics val chiqishlari)

Natijalar (results/graphs/, DPI=300):
    loss.png, metrics.png,
    precision.png, recall.png, map50.png, map5095.png, f1.png,
    confusion_matrix.png, confusion_matrix_normalized.png,
    PR_curve.png, P_curve.png, R_curve.png, F1_curve.png

Ishga tushirish:
    python generate_graphs.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import ensure_dirs

DPI = 300


def normalize_cols(df):
    """Ultralytics 8.4+ ustunlarida '(B)' qo'shimchasi bo'ladi (masalan metrics/precision(B)).
    Shu qo'shimchalarni olib tashlab turli versiyalar bilan ishlaydigan nomlar beradi."""
    cols = {}
    for c in df.columns:
        base = c.strip()
        for suf in ("(B)", "(M)", "(S)"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        cols[base] = c
    return cols


def col(df, mapping, name):
    return mapping.get(name)


def compute_f1(df, mapping):
    """P/R ustunlaridan F1 hisoblab beradi yoki yo'q bo'lsa Numpy array qaytaradi."""
    pc = col(df, mapping, "metrics/precision")
    rc = col(df, mapping, "metrics/recall")
    import numpy as np
    if pc and rc:
        p = df[pc].to_numpy(float)
        r = df[rc].to_numpy(float)
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
        return f1, (pc, rc)
    return None, None


def style(ax, title=None):
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(title or "", fontsize=11)
    ax.tick_params(labelsize=9)


def main() -> int:
    ensure_dirs()
    graphs = config.GRAPHS_DIR
    results_csv = config.EXP_DIR / "results.csv"
    if not results_csv.exists():
        results_csv = config.RESULTS_DIR / "training" / "results.csv"
    if not results_csv.exists():
        print("[XATO] results.csv topilmadi. Avval `train.py` ishga tushiring.", file=sys.stderr)
        return 1

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    ep = df["epoch"].astype(int)
    mapping = normalize_cols(df)

    # ── A. Loss grafigi ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    pairs = [("train/box_loss", "val/box_loss", "Box Loss"),
             ("train/cls_loss", "val/cls_loss", "Classification Loss"),
             ("train/dfl_loss", "val/dfl_loss", "DFL Loss")]
    colors = [("tab:red", "tab:orange"), ("tab:blue", "tab:cyan"), ("tab:green", "tab:olive")]
    for (tr, va, label), (c1, c2) in zip(pairs, colors):
        tr_c, va_c = col(df, mapping, tr), col(df, mapping, va)
        if tr_c:
            ax.plot(ep, df[tr_c], color=c1, lw=1.5, label=f"{label} (train)")
        if va_c:
            ax.plot(ep, df[va_c], color=c2, lw=1.5, ls="--", label=f"{label} (val)")
    style(ax, "Training / Validation Losses")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(graphs / "loss.png", dpi=DPI); plt.close(fig)

    # ── B. Metriklar grafigi ───────────────────────────────────
    f1_vals, f1_srcs = compute_f1(df, mapping)
    pl = {c: col(df, mapping, c) for c in
          ("metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95")}
    metric_cols = [(pl["metrics/precision"], "Precision", "tab:blue"),
                   (pl["metrics/recall"], "Recall", "tab:green"),
                   (pl["metrics/mAP50"], "mAP@0.5", "tab:red"),
                   (pl["metrics/mAP50-95"], "mAP@0.5:0.95", "tab:purple")]
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for c_val, label, color in metric_cols:
        if c_val is not None:
            ax2.plot(ep, df[c_val], color=color, lw=1.6, label=label)
    if f1_vals is not None:
        ax2.plot(ep, f1_vals, color="tab:orange", lw=1.6, ls="--", label="F1-score (hisoblangan)")
    style(ax2, "Validation Metrics per Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.legend(fontsize=9)
    fig2.tight_layout(); fig2.savefig(graphs / "metrics.png", dpi=DPI); plt.close(fig2)

    # ── Alohida metrikalar ─────────────────────────────────────
    single = [("metrics/precision", "precision.png", "Precision", pl["metrics/precision"]),
              ("metrics/recall", "recall.png", "Recall", pl["metrics/recall"]),
              ("metrics/mAP50", "map50.png", "mAP@0.5", pl["metrics/mAP50"]),
              ("metrics/mAP50-95", "map5095.png", "mAP@0.5:0.95", pl["metrics/mAP50-95"]),
              ("f1", "f1.png", "F1-score", "f1")]
    for _, fname, title, cc in single:
        if cc == "f1":
            if f1_vals is None:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(ep, f1_vals, lw=1.8, color="tab:blue")
            style(ax, title)
            ax.set_xlabel("Epoch"); ax.set_ylabel(title)
            ax.set_ylim(0, 1.05)
            fig.tight_layout(); fig.savefig(graphs / fname, dpi=DPI); plt.close(fig)
        elif cc is not None:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(ep, df[cc], lw=1.8, color="tab:blue")
            style(ax, title)
            ax.set_xlabel("Epoch"); ax.set_ylabel(title)
            ax.set_ylim(0, 1.05)
            fig.tight_layout(); fig.savefig(graphs / fname, dpi=DPI); plt.close(fig)

    # ── C..F. Ultralytics val chiqishlarini nusxalash ─────────
    val_dir = config.RESULTS_DIR / "validation" / "val"
    copies = {
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "PR_curve.png": "PR_curve.png",
        "P_curve.png": "P_curve.png",
        "R_curve.png": "R_curve.png",
        "F1_curve.png": "F1_curve.png",
        "BoxPR_curve.png": "PR_curve.png",
        "BoxP_curve.png": "P_curve.png",
        "BoxR_curve.png": "R_curve.png",
        "BoxF1_curve.png": "F1_curve.png",
    }
    if val_dir.exists():
        for src, dst in copies.items():
            s = val_dir / src
            if s.exists():
                import shutil
                shutil.copy2(s, graphs / dst)
                print(f"  ko'chirildi: {dst}")
            else:
                print(f"""[OGOH] topilmadi: {s}""")
    else:
        print(f"[OGOH] Ultralytics val papkasi topilmadi: {val_dir}. "
              f"Avval `validate.py` ishga tushiring.")

    print(f"Grafiklar saqlandi: {graphs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())