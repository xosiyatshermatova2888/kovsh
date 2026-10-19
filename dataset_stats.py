"""VAZIFA 1: Dataset statistikasi — har bir bolim (train/valid/test) uchun
rasmlar soni va har bir klass bo'yicha instansiyalar soni.

data.yaml topilmasa, standart data/{train,val,test}/images|labels tuzilmasi
ishlatiladi. Instansiyalar labels papkasidagi .txt fayllarni o'qib hisoblanadi
(har bir qatorning birinchi raqami klass ID'si - normativ sanaladi).
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

CLASS_NAMES = [
    "bucket_digging", "bucket_empty", "bucket_loaded", "dumping",
    "tooth", "truck_empty", "truck_loading",
]
CLASS_IDS = range(1, 8)
NAME_BY_ID = dict(zip(CLASS_IDS, CLASS_NAMES))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_yaml() -> Path | None:
    yamls = sorted({*BASE.glob("data.yaml"), *BASE.glob("*.yaml"),
                    *DATA.glob("*.yaml")})
    return yamls[0] if yamls else None


def split_paths_from_yaml(yaml_path: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    out = {}
    base = cfg.get("path", "")
    for key in ("train", "val", "test"):
        raw = cfg.get(key)
        if raw:
            p = Path(raw)
            if not p.exists() and base:
                p = Path(base) / p
            out[key] = p
    return out


def resolve_split(yaml_path: Path | None, key: str) -> Path:
    if yaml_path:
        p = split_paths_from_yaml(yaml_path).get(key)
        if p and p.exists():
            return p
    for c in (DATA / key / "images", BASE / key / "images"):
        if c.exists():
            return c
    return DATA / key / "images"


def count_labels(label_dir: Path) -> dict:
    counts = {}
    if not label_dir.exists():
        return counts
    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def main():
    yaml_path = find_data_yaml()
    if yaml_path:
        print("data.yaml topildi: {}\n".format(yaml_path))
    else:
        print("data.yaml topilmadi — standart data/ tuzilmasi ishlatiladi.\n")

    bolim = "Bolim"
    header = "{:<10}{:<9}".format(bolim, "Rasmlar") + "".join(
        "{:<16}".format(n) for n in CLASS_NAMES) + "{:<10}".format("Boshqa")
    print(header)
    print("-" * len(header))

    totals = {i: 0 for i in range(0, 10)}
    total_images = 0
    csv_rows = []
    for key, label in (("train", "Oquv"), ("val", "Tekshiruv"),
                       ("test", "Test")):
        img_dir = resolve_split(yaml_path, key)
        lb_dir = img_dir.parent / "labels"
        n_images = len([p for p in img_dir.iterdir()
                        if p.suffix.lower() in IMG_EXTS]) if img_dir.exists() else 0
        counts = count_labels(lb_dir)
        row = []
        for i in CLASS_IDS:
            c = counts.get(i, 0)
            totals[i] += c
            row.append(c)
        other = sum(v for k, v in counts.items() if k not in NAME_BY_ID)
        totals[9] += other
        total_images += n_images
        print("{:<10}{:<9}".format(label, n_images) + "".join(
            "{:<16}".format(c) for c in row) + "{:<10}".format(other))
        csv_rows.append([key, n_images] + row + [other])

    print("-" * len(header))
    csv_rows.append(["JAMI", total_images] + [totals[i] for i in CLASS_IDS] +
                    [totals[9]])
    print("{:<10}{:<9}".format("JAMI", total_images) + "".join(
        "{:<16}".format(totals[i]) for i in CLASS_IDS) +
        "{:<10}".format(totals[9]))

    total_inst = sum(totals.values())
    print("\nJami instansiyalar: {}".format(total_inst))
    print("Jami rasmlar: {}".format(total_images))

    out_csv = BASE / "dataset_statistics.csv"
    import pandas as pd

    pd.DataFrame(csv_rows, columns=["bolim", "rasmlar", *CLASS_NAMES,
                                    "boshqa"]).to_csv(out_csv, index=False,
                                                      encoding="utf-8-sig")
    print("CSV saqlandi: {}".format(out_csv))


if __name__ == "__main__":
    main()