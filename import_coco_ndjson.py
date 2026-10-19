"""Roboflow/Ultralytics COC-NDJSON eksportini YOLO formatiga aylantiradi.

Kirish:
    kovsh-motionv6icoco.ndjson  (har qator: {"type":"dataset"|"image", ...})

Jarayon:
    1. `dataset` qatoridan class_names ni o'qiydi.
    2. Eski 2-class data/ kontentini data_old_2cls/ ga ko'chiradi.
    3. Har `image` qatoridagi url'dan rasmni yuklab oladi.
    4. Har rasm uchun YOLO label faylini yozadi ("cls cx cy w h" — allaqachon normalize).
    5. data/data.yaml ni hosil qiladi.

Ishga tushirish:
    python import_coco_ndjson.py
"""

import concurrent.futures as cf
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parent
NDJSON = BASE_DIR / "kovsh-motionv6icoco.ndjson"
DATA_DIR = BASE_DIR / "data"
OLD_DIR = BASE_DIR / "data_old_2cls"
WORKERS = 12
MAX_RETRIES = 3
UA = {"User-Agent": "Mozilla/5.0 (kovsh-dataset-importer)"}

failures = []


def download_file(url: str, dest: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                status = r.status
            if not data or status != 200:
                raise RuntimeError(f"status={status}")
            dest.write_bytes(data)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES:
                failures.append(dest.name)
                print(f"    [XATO] {dest.name}: {e}")
                return False
            time.sleep(1.5 * attempt)
    return False


def process_image(o: dict, class_names: dict) -> None:
    split = o["split"]
    fname = o["file"]
    stem = Path(fname).stem
    img_path = DATA_DIR / split / "images" / fname
    lbl_path = DATA_DIR / split / "labels" / f"{stem}.txt"

    if not img_path.exists() and not download_file(o["url"], img_path):
        return

    lines = []
    for box in o.get("annotations", {}).get("boxes", []):
        if len(box) != 5:
            continue
        cls, cx, cy, w, h = box
        if cls not in class_names:
            continue
        cx = max(0.0, min(1.0, float(cx)))
        cy = max(0.0, min(1.0, float(cy)))
        w = max(0.0, min(1.0, float(w)))
        h = max(0.0, min(1.0, float(h)))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def backup_old_data() -> None:
    """Eski 2-class datasetni alohida papkaga ko'chiradi."""
    if not DATA_DIR.exists():
        return
    has_old = any((DATA_DIR / s / "images").exists() for s in ("train", "val", "test"))
    if not has_old:
        return
    if OLD_DIR.exists():
        shutil.rmtree(OLD_DIR)
    OLD_DIR.mkdir(parents=True, exist_ok=True)
    for item in DATA_DIR.iterdir():
        if item.name in ("train", "val", "test", "data.yaml"):
            shutil.move(str(item), str(OLD_DIR / item.name))
    print(f"  eski dataset -> {OLD_DIR}")


def main() -> int:
    if not NDJSON.exists():
        print(f"[XATO] {NDJSON} topilmadi.", file=sys.stderr)
        return 1

    class_names = None
    images = []
    with NDJSON.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["type"] == "dataset":
                class_names = {int(k): v for k, v in o["class_names"].items()}
            elif o["type"] == "image":
                images.append(o)

    if class_names is None:
        print("[XATO] dataset qatori topilmadi.", file=sys.stderr)
        return 1

    nc = len(class_names)
    print(f"[1/3] class_names ({nc}): {class_names}")
    print(f"      rasmlar: {len(images)}")
    for split in sorted({o["split"] for o in images}):
        print(f"      {split}: {sum(1 for o in images if o['split'] == split)}")

    print("[2/3] eski datasetni zaxiralash")
    backup_old_data()

    for split in {o["split"] for o in images}:
        for sub in ("images", "labels"):
            (DATA_DIR / split / sub).mkdir(parents=True, exist_ok=True)

    print(f"[3/3] {len(images)} ta rasm yuklanmoqda (workers={WORKERS})")
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(process_image, o, class_names) for o in images]
        for fut in cf.as_completed(futures):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(futures):
                print(f"  ... {done}/{len(futures)}  ({time.time()-t0:.0f}s)")

    # data.yaml
    names = [class_names[i] for i in range(nc)]
    yaml_content = {
        "path": DATA_DIR.as_posix(),
        "train": (DATA_DIR / "train" / "images").as_posix(),
        "val": (DATA_DIR / "val" / "images").as_posix(),
        "test": (DATA_DIR / "test" / "images").as_posix(),
        "nc": nc,
        "names": names,
    }
    (DATA_DIR / "data.yaml").write_text(
        yaml.dump(yaml_content, sort_keys=False), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("IMPORT XISOBOTI")
    print("=" * 60)
    total_boxes = 0
    for split in sorted({o["split"] for o in images}):
        n_imgs = len(list((DATA_DIR / split / "images").glob("*")))
        n_lbl = len(list((DATA_DIR / split / "labels").glob("*.txt")))
        boxes = sum(
            len((DATA_DIR / split / "labels" / p.name).read_text().splitlines())
            for p in (DATA_DIR / split / "labels").glob("*.txt")
        )
        total_boxes += boxes
        print(f"  {split:6s}: images={n_imgs:4d} labels={n_lbl:4d} boxes={boxes:5d}")
    print(f"  jami box: {total_boxes}")
    print(f"  yuklashda xato: {len(failures)}")
    if failures:
        for name in failures[:10]:
            print(f"    - {name}")
    print(f"  vaqt: {time.time()-t0:.0f}s")
    print(f"  data.yaml: {DATA_DIR/'data.yaml'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
