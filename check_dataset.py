"""Datasetni tekshiradi va hisobot beradi.

- data.yaml ni o'qiydi va yo'llarni himoya qiladi
- har splitdagi rasm/label sonlarini hisoblaydi
- yopishib qolgan / bo'sh / noto'g'ri label fayllarini topadi
- har bir class bo'yicha label sonini chiqaradi

Ishga tushirish:
    python check_dataset.py
"""

import sys
from collections import Counter
from pathlib import Path

import yaml

# Windows konsolida UTF-8 belgilarni chiqarish uchun
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import NC, ensure_dirs


def validate_label_file(path: Path, name: str) -> tuple:
    """Label faylni tekshiradi -> (xatolar_soni, classlar_Counter)."""
    errors = 0
    classes = Counter()
    if not path.exists():
        return 1, classes
    content = path.read_text().strip()
    if not content:
        return 1, classes
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 5:
            errors += 1
            continue
        cls = int(parts[0])
        if cls < 0 or cls >= NC:
            errors += 1
            continue
        vals = [float(p) for p in parts[1:5]]
        # YOLO: 0 <= cx,cy <= 1 va 0 < w,h <= 1
        if any(v < 0.0 or v > 1.0 for v in vals):
            errors += 1
            continue
        classes[cls] += 1
    return errors, classes


def main() -> int:
    ensure_dirs()

    # ── data.yaml ──────────────────────────────────────────────
    print("─" * 62)
    print("data.yaml tekshiruvi")
    print("─" * 62)
    if not config.DATA_YAML.exists():
        print(f"[XATO] {config.DATA_YAML} topilmadi. Avval `prepare_dataset.py` ishga tushiring.")
        return 1
    cfg = yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8"))
    print(f"  fayl: {config.DATA_YAML}")
    print(f"  nc:   {cfg.get('nc')}")
    print(f"  names:{cfg.get('names')}")
    if cfg.get("nc") != NC or list(cfg.get("names", [])) != config.CLASS_NAMES:
        print(f"[OGOH] names/nc data.yaml ichida maqsadli classlardan farq qiladi: "
              f"{cfg.get('names')}")

    ok = True
    total_classes = Counter()
    total_images = 0
    total_labels = 0
    empty_files = 0
    bad_files = 0
    missing_labels = 0
    dims_ok = True

    for split in ("train", "val", "test"):
        print()
        print(f"── split: {split} ──")
        cfg_key = cfg.get(split)
        idir = Path(cfg_key) if cfg_key else None
        if idir is None or not idir.exists():
            print(f"  [OGOH] '{split}' papka topilmadi: {cfg.get(split)}")
            ok = False
            continue
        ldir = idir.parent / "labels"
        imgs = sorted([p for p in idir.glob("*") if p.suffix.lower() in
                       (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
        lbls = sorted(ldir.glob("*.txt"))
        print(f"  images: {len(imgs)}  labels: {len(lbls)}")
        per = Counter()
        for im in imgs:
            stem = im.stem
            lf = ldir / f"{stem}.txt"
            err, cls_c = validate_label_file(lf, split)
            per.update(cls_c)
            if not lf.exists():
                missing_labels += 1
            elif not lf.read_text().strip():
                empty_files += 1
            if err:
                bad_files += 1
                if bad_files <= 5:
                    print(f"    [XATO] {lf.name}: {err} xato satr")
        for cls_i in range(NC):
            print(f"  {config.CLASS_NAMES[cls_i]:16s} labels: {per.get(cls_i, 0)}")
        total_classes.update(per)
        total_images += len(imgs)
        total_labels += len(lbls)

        # rasm hajmi tekshiruvi (birinchi 3 tasi)
        for im in imgs[:3]:
            import cv2
            img = cv2.imread(str(im))
            if img is None:
                print(f"    [XATO] o'qib bo'lmadi: {im.name}")
                dims_ok = False

    print()
    print("═" * 62)
    print("UMUMIY XISOBOT")
    print("═" * 62)
    print(f"Jami rasmlar:      {total_images}")
    print(f"Jami label fayllar:{total_labels}")
    for cls_i in range(NC):
        print(f"{config.CLASS_NAMES[cls_i]:16s} box:  {total_classes.get(cls_i, 0)}")
    print(f"bo'sh label:       {empty_files}")
    print(f"missing label:     {missing_labels}")
    print(f"noto'g'ri label:   {bad_files}")
    print(f"rasm o'qish:       {'OK' if dims_ok else 'XATO bor'}")
    print()
    if not ok or missing_labels or empty_files or bad_files or not dims_ok:
        print(">>> NATIJA: datasetda xatoliklar bor, trainingdan oldin tuzatish kerak.")
        return 1
    print(">>> NATIJA: dataset tayyor, training boshlash mumkin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())