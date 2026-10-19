"""Video'dan YOLO train/test datasetini avtomatik yaratadi.

Ish tartibi:
  1. `SOURCE_VIDEO` faylidan barcha kadrlarni ajratadi.
  2. `SOURCE_MODEL` (avtomatik labeling) bilan har kadrda obyektlarni aniqlaydi.
  3. Classlarni 2 ta maqsadli classga moslashtiradi:
        exp-2 class 0,1 -> bucket-teeth(0)
        exp-2 class 3   -> truck(1)
        exp-2 class 2   -> stone (tashlab yuboriladi)
  4. Kamida 1 ta maqsadli deteksiya bor kadrlarni train/val/test ga ajratadi
     (truck bor kadrlar bo'yicha stratifikatsiya qilinadi).
  5. `data/data.yaml` ni hosil qiladi.

Natijalar:
    data/train/images, data/train/labels
    data/val/images,   data/val/labels
    data/test/images,  data/test/labels
    data/data.yaml

Ishga tushirish:
    python prepare_dataset.py
"""

import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import ensure_dirs


CLASS_MAP = {0: 0, 1: 0, 2: None, 3: 1}  # exp-2 cls -> target cls (None=tashla)


def extract_frames(video_path: Path, out_dir: Path) -> list:
    """Videoni kadrlarga ajratadi, (index, path) ro'yxatini qaytaradi."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Videoni ochib bo'lmadi: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        path = out_dir / f"frame_{idx + 1:04d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(path)
        idx += 1
    cap.release()
    return frames


def main() -> int:
    from ultralytics import YOLO

    ensure_dirs()
    src_model = config.SOURCE_MODEL
    src_video = config.SOURCE_VIDEO
    if not src_model.exists():
        print(f"[XATO] Avto-labeling modeli topilmadi: {src_model}", file=sys.stderr)
        return 1
    if not src_video.exists():
        print(f"[XATO] Video topilmadi: {src_video}", file=sys.stderr)
        return 1

    tmp_dir = config.DATA_DIR / "_video_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"[1/5] Kadrlarni ajratish: {src_video}")
    frames = extract_frames(src_video, tmp_dir)
    print(f"      {len(frames)} ta kadr")

    print(f"[2/5] Model yuklanmoqda: {src_model}")
    model = YOLO(str(src_model))
    print(f"      model classlari: {model.names}")

    print(f"[3/5] Avto-labeling (conf={config.LABEL_CONF}, imgsz={config.LABEL_IMGSZ})")
    # {frame_stem: {'objects': [(target_cls, cx,cy,w,h)*]} , 'has_truck': bool}
    labeled = {}
    total_det = 0
    dropped_stone = 0
    empty_target = 0
    for path in frames:
        r = model.predict(str(path), conf=config.LABEL_CONF, imgsz=config.LABEL_IMGSZ,
                          device=config.DEVICE, verbose=False)[0]
        objs = []
        has_truck = False
        if r.boxes is not None and len(r.boxes):
            for box in r.boxes:
                src_cls = int(box.cls[0])
                tgt = CLASS_MAP.get(src_cls)
                if tgt is None:
                    if src_cls == 2:
                        dropped_stone += 1
                    continue
                cx, cy, w, h = box.xywhn[0].tolist()
                objs.append((tgt, cx, cy, w, h))
                if tgt == 1:
                    has_truck = True
        total_det += len(objs)
        if not objs:
            empty_target += 1
        else:
            labeled[path.stem] = {"path": path, "objs": objs, "has_truck": has_truck}

    print(f"      maqsadli deteksiya: {total_det}, stone tashlandi: {dropped_stone},"
          f" labelsiz kadr: {empty_target}")

    # Stratifikatsiya: truck bor/yo'q guruhlar ichidan random taqsimot
    rng = random.Random(config.SEED)
    with_truck = [v for v in labeled.values() if v["has_truck"]]
    without = [v for v in labeled.values() if not v["has_truck"]]

    def split(items, val_frac, test_frac):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test = items[:n_test]
        val = items[n_test:n_test + n_val]
        train = items[n_test + n_val:]
        return train, val, test

    tr_truck, va_truck, te_truck = split(with_truck, config.VAL_SPLIT, config.TEST_SPLIT)
    tr_no, va_no, te_no = split(without, config.VAL_SPLIT, config.TEST_SPLIT)

    splits = {
        "train": (tr_truck + tr_no),
        "val": (va_truck + va_no),
        "test": (te_truck + te_no),
    }

    print(f"[4/5] Dataset yozilmoqda (train/val/test = "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])})")
    img_dirs = {}
    lbl_dirs = {}
    for split in splits:
        idir = config.DATA_DIR / split / "images"
        ldir = config.DATA_DIR / split / "labels"
        for d in (idir, ldir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        img_dirs[split] = idir
        lbl_dirs[split] = ldir

    per_split_class = {s: {0: 0, 1: 0} for s in splits}
    train_idx = count_saved = 0
    for split, items in splits.items():
        for item in items:
            stem = item["path"].stem
            img_path = img_dirs[split] / f"{stem}.jpg"
            shutil.copy2(item["path"], img_path)
            label_file = lbl_dirs[split] / f"{stem}.txt"
            with open(label_file, "w") as f:
                for (tgt, cx, cy, w, h) in item["objs"]:
                    f.write(f"{tgt} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    per_split_class[split][tgt] += 1
            count_saved += 1

    print(f"[5/5] data.yaml yozilmoqda")
    yaml_content = (
        f"# Avtomatik yaratildi: prepare_dataset.py from {src_video.name}\n"
        f"path: {config.DATA_DIR.as_posix()}\n"
        f"train: {str(img_dirs['train']).replace(chr(92), '/')}\n"
        f"val: {str(img_dirs['val']).replace(chr(92), '/')}\n"
        f"test: {str(img_dirs['test']).replace(chr(92), '/')}\n"
        f"nc: {config.NC}\n"
        f"names: {config.CLASS_NAMES}\n"
    )
    config.DATA_YAML.write_text(yaml_content, encoding="utf-8")

    print()
    print("=" * 60)
    print("DATASET XISOBOTI")
    print("=" * 60)
    print(f"Umumiy kadr (video):        {len(frames)}")
    print(f"Saqlangan rasm (>=1 object):{count_saved}")
    print(f"train/val/test:             "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    for split, cnt in per_split_class.items():
        print(f"  {split:6s}: bucket-teeth={cnt[0]:4d}, truck={cnt[1]:4d}")
    print(f"data.yaml: {config.DATA_YAML}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())