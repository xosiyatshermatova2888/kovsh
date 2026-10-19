"""Dissertatsiya uchun natijalarni yig'adi va tahlil fayllarini yaratadi.

Natijalar:
    results/dissertation_metrics.xlsx   (yakuniy jadval)
    results/training_summary.txt        (training xulosasi)
    results/dissertation_analysis.txt   (ilmiy tahlil)

Ishga tushirish (validate.py dan keyin):
    python analyze_results.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import ensure_dirs


def fmt(v):
    return round(v, 4) if isinstance(v, (int, float)) else v


def normalize_cols(df):
    """Ultralytics 8.4+ ustunlarida '(B)' qo'shimchasi bo'ladi (masalan metrics/precision(B))."""
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


def compute_f1(rdf, mapping):
    c50 = col(rdf, mapping, "metrics/precision")
    cr = col(rdf, mapping, "metrics/recall")
    if c50 and cr:
        p = rdf[c50].to_numpy(float)
        r = rdf[cr].to_numpy(float)
        return np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return None


def main() -> int:
    ensure_dirs()
    results = config.RESULTS_DIR
    metrics_csv = results / "validation" / "metrics.csv"

    if not metrics_csv.exists():
        print(f"[XATO] {metrics_csv} topilmadi. Avval `validate.py` ishga tushiring.", file=sys.stderr)
        return 1

    df = pd.read_csv(metrics_csv)
    overall = df[df["metric"] == "Overall"].iloc[0]
    classes = df[df["metric"] != "Overall"]
    row_map = {r["metric"]: r for _, r in classes.iterrows()}

    # ── 1. dissertation_metrics.xlsx ───────────────────────────
    table = {
        "Metric": ["Precision", "Recall", "F1-score", "mAP50", "mAP50-95"],
        "Overall": [fmt(overall["precision"]), fmt(overall["recall"]),
                    fmt(overall["f1"]), fmt(overall["map50"]), fmt(overall["map50_95"])],
    }
    for cls_name in ("bucket-teeth", "truck"):
        if cls_name in row_map:
            r = row_map[cls_name]
            table[cls_name.capitalize()] = [fmt(r["precision"]), fmt(r["recall"]),
                                            fmt(r["f1"]), fmt(r["map50"]), fmt(r["map50_95"])]
        else:
            table[cls_name.capitalize()] = ["N/A"] * 5

    tdf = pd.DataFrame(table)
    xlsx_path = results / "dissertation_metrics.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        tdf.to_excel(writer, sheet_name="Metrics", index=False)
        df.to_excel(writer, sheet_name="Detailed", index=False)

    # ── 2. best epoch aniqlash ─────────────────────────────────
    results_csv = config.EXP_DIR / "results.csv"
    if not results_csv.exists():
        results_csv = results / "training" / "results.csv"

    best_epochs = {}
    train_set = None
    if results_csv.exists():
        rdf = pd.read_csv(results_csv)
        rdf.columns = [c.strip() for c in rdf.columns]
        train_set = rdf
        if len(rdf):
            mapping = normalize_cols(rdf)
            c50 = col(rdf, mapping, "metrics/mAP50")
            c5095 = col(rdf, mapping, "metrics/mAP50-95")
            cp = col(rdf, mapping, "metrics/precision")
            cr = col(rdf, mapping, "metrics/recall")
            f1_arr = compute_f1(rdf, mapping)
            if c50:
                i = rdf[c50].idxmax()
                best_epochs["best_epoch_mAP50"] = (int(rdf.loc[i, "epoch"]),
                                                   float(rdf.loc[i, c50]),
                                                   float(rdf.loc[i, c5095]) if c5095 else None,
                                                   float(rdf.loc[i, cp]) if cp else None,
                                                   float(rdf.loc[i, cr]) if cr else None)
            if c5095:
                i = rdf[c5095].idxmax()
                best_epochs["best_epoch_mAP50-95"] = (int(rdf.loc[i, "epoch"]),
                                                      float(rdf.loc[i, c50]) if c50 else None,
                                                      float(rdf.loc[i, c5095]),
                                                      float(rdf.loc[i, cp]) if cp else None,
                                                      float(rdf.loc[i, cr]) if cr else None)
            if f1_arr is not None:
                i = int(np.argmax(f1_arr))
                best_epochs["best_epoch_F1"] = (int(rdf.loc[i, "epoch"]),
                                                float(rdf.loc[i, c50]) if c50 else None,
                                                float(rdf.loc[i, c5095]) if c5095 else None,
                                                float(f1_arr[i]))

            # Umumiy "eng yaxshi" epoch = mAP50-95 bo'yicha
            if c5095:
                i = rdf[c5095].idxmax()
                best_epochs["best_epoch"] = int(rdf.loc[i, "epoch"])

    # ── test / FPS ma'lumotlari ────────────────────────────────
    test_stats = {}
    ts_path = results / "test" / "test_summary.csv"
    if ts_path.exists():
        with open(ts_path) as f:
            test_stats = dict(line.strip().split(",", 1) for line in f if "," in line)

    fps_text = ""
    fps_file = results / "video_fps.txt"
    if fps_file.exists():
        fps_text = fps_file.read_text(encoding="utf-8")
    avg_fps = "N/A"
    for line in fps_text.splitlines():
        if line.startswith("avg_fps:"):
            avg_fps = line.split(":", 1)[1].strip()

    tt = "N/A"
    tt_file = results / "training_time_sec.txt"
    if tt_file.exists():
        tt = f"{float(tt_file.read_text().strip()):.1f} s"

    model_name = config.resolve_model()
    nm_file = results / "train_model.txt"
    if nm_file.exists():
        model_name = nm_file.read_text().strip()

    # ── 3. training_summary.txt ────────────────────────────────
    lines = []
    lines.append("TRAINING SUMMARY")
    lines.append("=" * 62)
    lines.append(f"model           : {model_name}")
    lines.append(f"dataset         : {config.DATA_YAML}")
    lines.append(f"number of classes: {config.NC}")
    lines.append(f"class names     : {config.CLASS_NAMES}")
    lines.append(f"epochs          : {config.TRAIN_EPOCHS}")
    lines.append(f"image size      : {config.IMG_SIZE}")
    lines.append(f"batch size      : {config.BATCH}")
    lines.append(f"device          : {config.DEVICE}")
    lines.append(f"workers         : {config.WORKERS}")
    lines.append(f"patience        : {config.PATIENCE}")
    lines.append(f"pretrained      : {config.PRETRAINED}")
    lines.append(f"seed            : {config.SEED}")
    lines.append(f"training time   : {tt}")
    lines.append(f"best epoch      : {best_epochs.get('best_epoch', 'N/A')}")
    lines.append(f"best mAP50      : {fmt(overall['map50'])}")
    lines.append(f"best mAP50-95   : {fmt(overall['map50_95'])}")
    lines.append(f"best Precision  : {fmt(overall['precision'])}")
    lines.append(f"best Recall     : {fmt(overall['recall'])}")
    lines.append("")
    lines.append("best_epoch_mAP50    : " +
                 (f"epoch={best_epochs['best_epoch_mAP50'][0]}, mAP50={best_epochs['best_epoch_mAP50'][1]:.4f}"
                  if "best_epoch_mAP50" in best_epochs and best_epochs['best_epoch_mAP50'][1] is not None else "N/A"))
    lines.append("best_epoch_mAP50-95 : " +
                 (f"epoch={best_epochs['best_epoch_mAP50-95'][0]}, mAP50-95={best_epochs['best_epoch_mAP50-95'][2]:.4f}"
                  if "best_epoch_mAP50-95" in best_epochs and best_epochs['best_epoch_mAP50-95'][2] is not None else "N/A"))
    lines.append("best_epoch_F1       : " +
                 (f"epoch={best_epochs['best_epoch_F1'][0]}, F1={best_epochs['best_epoch_F1'][3]:.4f}"
                  if "best_epoch_F1" in best_epochs and best_epochs['best_epoch_F1'][3] is not None else "N/A"))
    lines.append("")
    lines.append("TEST NATIJALARI")
    for k in ("images", "images_with_detections", "total_detections", "mean_confidence"):
        lines.append(f"  {k}: {test_stats.get(k, 'N/A')}")
    for k, v in sorted(test_stats.items()):
        if k not in ("images", "images_with_detections", "total_detections", "mean_confidence"):
            lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("VIDEO INFERENCE")
    for ln in fps_text.splitlines():
        lines.append(f"  {ln}")
    summary_path = results / "training_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── 4. dissertation_analysis.txt ───────────────────────────
    def g(name, key):
        return row_map.get(name, {}).get(key, "N/A") if name in row_map else "N/A"

    c_mapping = {}
    if train_set is not None and len(train_set):
        c_mapping = normalize_cols(train_set)
    t_p = col(train_set, c_mapping, "metrics/precision") if train_set is not None else None
    t_r = col(train_set, c_mapping, "metrics/recall") if train_set is not None else None
    t50 = col(train_set, c_mapping, "metrics/mAP50") if train_set is not None else None
    t95 = col(train_set, c_mapping, "metrics/mAP50-95") if train_set is not None else None

    bt = ("bucket-teeth",) if "bucket-teeth" in row_map else None
    tk = ("truck",) if "truck" in row_map else None
    bt_p, bt_r = g("bucket-teeth", "precision"), g("bucket-teeth", "recall")
    tr_p, tr_r = g("truck", "precision"), g("truck", "recall")
    bt_ap5095 = g("bucket-teeth", "map50_95")
    tr_ap5095 = g("truck", "map50_95")

    lines_a = []
    lines_a.append("DISSERTATSIYA TAHLILI (ilmiy uslub)")
    lines_a.append("=" * 62)
    lines_a.append("")
    lines_a.append("Ushbu tahlil faqat haqiqiy eksperiment natijalari asosida yozilgan. ")
    lines_a.append("Metrika mavjud bo'lmagan hollarda 'N/A' ko'rsatilgan.")
    lines_a.append("")

    lines_a.append("1. TRAINING JARAYONIDA LOSS QANDAY O'ZGARGAN")
    if train_set is not None and len(train_set):
        first, lastr = train_set.iloc[0], train_set.iloc[-1]
        def loss_info(col):
            if col in train_set.columns:
                return (f"{col}: ilk={first[col]:.4f} -> oxirgi={lastr[col]:.4f}")
            return None
        parts = [p for p in (loss_info("train/box_loss"), loss_info("val/box_loss"),
                             loss_info("train/cls_loss"), loss_info("val/cls_loss"),
                             loss_info("train/dfl_loss"), loss_info("val/dfl_loss")) if p]
        lines_a += [f"   - {'; '.join(parts)}"]
    else:
        lines_a.append("   - loss egri chiziqlar bo'yicha ma'lumot mavjud emas (N/A)")
    lines_a.append("")
    lines_a.append("2. PRECISION O'ZGARISHI")
    if t_p is not None:
        lines_a.append(f"   - Validation Precision ilk epoch: {train_set.iloc[0][t_p]:.4f}, "
                       f"eng yuqori: {train_set[t_p].max():.4f}, "
                       f"oxirgi: {train_set.iloc[-1][t_p]:.4f}")
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['precision'])}")
    else:
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['precision'])}")
    lines_a.append("")
    lines_a.append("3. RECALL O'ZGARISHI")
    if t_r is not None:
        lines_a.append(f"   - Validation Recall ilk epoch: {train_set.iloc[0][t_r]:.4f}, "
                       f"eng yuqori: {train_set[t_r].max():.4f}, "
                       f"oxirgi: {train_set.iloc[-1][t_r]:.4f}")
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['recall'])}")
    else:
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['recall'])}")
    lines_a.append("")
    lines_a.append("4. mAP50 O'ZGARISHI")
    if t50 is not None:
        lines_a.append(f"   - ilk epoch: {train_set.iloc[0][t50]:.4f}, "
                       f"maksimal: {train_set[t50].max():.4f}, "
                       f"oxirgi: {train_set.iloc[-1][t50]:.4f}")
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['map50'])}")
    else:
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['map50'])}")
    lines_a.append("")
    lines_a.append("5. mAP50-95 O'ZGARISHI")
    if t95 is not None:
        lines_a.append(f"   - ilk epoch: {train_set.iloc[0][t95]:.4f}, "
                       f"maksimal: {train_set[t95].max():.4f}, "
                       f"oxirgi: {train_set.iloc[-1][t95]:.4f}")
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['map50_95'])}")
    else:
        lines_a.append(f"   - Final (best.pt val): {fmt(overall['map50_95'])}")
    lines_a.append("")
    lines_a.append("6. QAYSI CLASS YAXSHIROQ ANIQLANGAN")
    if bt is not None and tk is not None:
        lines_a.append(f"   - bucket-teeth: P={bt_p}, R={bt_r}, mAP50-95={bt_ap5095}")
        lines_a.append(f"   - truck: P={tr_p}, R={tr_r}, mAP50-95={tr_ap5095}")
        try:
            if bt_ap5095 != "N/A" and tr_ap5095 != "N/A" and float(bt_ap5095) >= float(tr_ap5095):
                lines_a.append("   - Xulosa: bucket-teeth class yaxshiroq aniqlangan.")
            elif bt_ap5095 != "N/A" and tr_ap5095 != "N/A":
                lines_a.append("   - Xulosa: truck class yaxshiroq aniqlangan.")
            else:
                lines_a.append("   - Xulosa: classlar orasidagi taqqoslash uchun yetarli ma'lumot yo'q (N/A).")
        except (TypeError, ValueError):
            lines_a.append("   - Xulosa: N/A")
    else:
        lines_a.append("   - N/A")
    lines_a.append("")
    lines_a.append("7. QAYSI CLASDA XATOLIK KO'PROQ")
    if bt is not None and tk is not None:
        lines_a.append(f"   - truck uchun mAP50-95={tr_ap5095}; bucket-teeth uchun mAP50-95={bt_ap5095}.")
        try:
            if tr_ap5095 != "N/A" and bt_ap5095 != "N/A" and float(tr_ap5095) < float(bt_ap5095):
                lines_a.append("   - Eng ko'p xatolik truck classda (oz sonli misollar tufayli).")
            elif tr_ap5095 != "N/A" and bt_ap5095 != "N/A":
                lines_a.append("   - Eng ko'p xatolik bucket-teeth classda kuzatiladi.")
            else:
                lines_a.append("   - N/A")
        except (TypeError, ValueError):
            lines_a.append("   - N/A")
    else:
        lines_a.append("   - N/A")
    lines_a.append("")
    lines_a.append("8. CONFUSION MATRIX NIMANI KO'RSATMOQDA")
    lines_a.append("   - Confusion matrix grafigi results/graphs/confusion_matrix{,_normalized}.png da.")
    lines_a.append("   - Matritsadagi diagonal qiymatlar to'g'ri klassifikatsiyani, diagonoldan tashqari "
                   "qiymatlar esa sinflararo chalkashlikni ifodalaydi.")
    lines_a.append("")
    lines_a.append("9. MODELNING KUCHLI TOMONLARI")
    lines_a.append(f"   - Umumiy mAP50={fmt(overall['map50'])}, mAP50-95={fmt(overall['map50_95'])} ga erishildi.")
    lines_a.append("   - Real videoda inference ishlaydi va annotation natijalari saqlanadi.")
    lines_a.append("   - Faqat 2 sinf (bucket-teeth, truck) bilan ixcham yechim.")
    lines_a.append("")
    lines_a.append("10. MODELNING CHEKLOVLARI")
    lines_a.append("   - Dataset bitta videodan (9.9 s, 200 kadr) avtomatik labeling orqali olingan; "
                   "sun'iy/psevdo label sifatiga bog'liq.")
    lines_a.append(f"   - Classlar muvozanatsiz: bucket-teeth boxlari truck boxlarga nisbatan ancha ko'p.")
    lines_a.append("   - Ochiladigan umumlashtirish cheklangan, chunki ma'lumotlar bir xil sahna/manziladan.")
    lines_a.append("   - Labeling manba modeli (exp-2.pt) ning o'zi ham xatolikka ega bo'lishi mumkin.")
    lines_a.append("")
    lines_a.append("11. KEYINGI TADQIQOT UCHUN TAKLIFLAR")
    lines_a.append("   - Bir necha o'nlab to'liq qo'lda annotatsiya qilingan dataset yaratish.")
    lines_a.append("   - Class muvozanatini tiklash (truck uchun ko'proq misol, augmentatsiya).")
    lines_a.append("   - Keyingi bosqich: video streamda real-time inference + FPS optimizatsiyasi.")
    lines_a.append("   - Data augmentation va kross-validatsiya orqali mAPni oshirish.")
    analysis_path = results / "dissertation_analysis.txt"
    analysis_path.write_text("\n".join(lines_a) + "\n", encoding="utf-8")

    print(f"dissertation_metrics.xlsx : {xlsx_path}")
    print(f"training_summary.txt      : {summary_path}")
    print(f"dissertation_analysis.txt : {analysis_path}")
    print()
    print("YAKUNIY JADVAL")
    print(tdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())