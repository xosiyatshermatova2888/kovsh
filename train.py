"""YOLO training skripti.

Ishga tushirish:
    python train.py

Model:
    - `config.MODEL_NAME` (default: yolo11m.pt) — internetdan yuklanadi.
    - Muqobil lokal modellar: `config.AVAILABLE_MODELS`.

Natija (Ultralytics):
    runs/teeth_truck_exp1/weights/best.pt
    runs/teeth_truck_exp1/weights/last.pt
    runs/teeth_truck_exp1/results.csv

Nusxalar `results/training/` ga ko'chiriladi.
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import ensure_dirs


def main() -> int:
    from ultralytics import YOLO

    ensure_dirs()

    data_yaml = config.DATA_YAML
    if not data_yaml.exists():
        print(f"[XATO] {data_yaml} topilmadi. Avval `check_dataset.py` natijasini ko'ring.", file=sys.stderr)
        return 1

    model_name = config.resolve_model()
    print("=" * 62)
    print("TRAINING BOSHLANMOQDA")
    print("=" * 62)
    print(f"model        : {model_name}")
    print(f"dataset      : {data_yaml}")
    print(f"epochs       : {config.TRAIN_EPOCHS}")
    print(f"imgsz        : {config.IMG_SIZE}")
    print(f"batch        : {config.BATCH}")
    print(f"device       : {config.DEVICE}")
    print(f"workers      : {config.WORKERS}")
    print(f"patience     : {config.PATIENCE}")
    print(f"pretrained   : {config.PRETRAINED}")
    print(f"seed         : {config.SEED}")
    print(f"output       : {config.EXP_DIR}")
    print("-" * 62)

    # Agar eksperiment avval ishlatilgan bo'lsa, ultralytics xato bermasligi uchun nomni alohidalash
    if config.EXP_DIR.exists() and (config.EXP_DIR / "weights" / "best.pt").exists():
        import datetime
        backup = config.RUNS_DIR / f"{config.PROJECT_NAME}_bak_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(str(config.EXP_DIR), str(backup))
        print(f"Eski eksperiment zaxira qilindi: {backup}")

    t0 = time.time()
    model = YOLO(model_name)

    try:
        model.train(
            data=str(data_yaml),
            epochs=config.TRAIN_EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH,
            device=config.DEVICE,
            workers=config.WORKERS,
            patience=config.PATIENCE,
            pretrained=config.PRETRAINED,
            seed=config.SEED,
            project=str(config.RUNS_DIR),
            name=config.PROJECT_NAME,
            exist_ok=True,
            **config.TRAIN_KWARGS,
        )
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
            print(f"[XATO] GPU xotirasi yetmayapti (batch={config.BATCH}). "
                  f"`config.py` da BATCH ni 4 yoki 2 qilib kamaytiring.", file=sys.stderr)
        raise

    train_time = time.time() - t0

    best = config.EXP_DIR / "weights" / "best.pt"
    last = config.EXP_DIR / "weights" / "last.pt"
    results_csv = config.EXP_DIR / "results.csv"

    if not best.exists() or not last.exists():
        print("[XATO] weights fayllari topilmadi — training muvaffaqiyatsiz yakunlandi.", file=sys.stderr)
        return 1

    # Nusxalar
    out_train = config.RESULTS_DIR / "training"
    out_train.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_train / "best.pt")
    shutil.copy2(last, out_train / "last.pt")
    if results_csv.exists():
        shutil.copy2(results_csv, out_train / "results.csv")
    (config.RESULTS_DIR / "training_time_sec.txt").write_text(f"{train_time:.1f}\n")
    (config.RESULTS_DIR / "train_model.txt").write_text(model_name + "\n")

    print("=" * 62)
    print("TRAINING YAKUNLANDI")
    print(f"training vaqti : {train_time:.1f} s")
    print(f"best.pt        : {best}")
    print(f"last.pt        : {last}")
    print(f"results.csv    : {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())