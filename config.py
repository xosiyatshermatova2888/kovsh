"""Markaziy konfiguratsiya fayli.

Barcha yo'llar, model, train va dataset parametrlari shu yerda
bitta joydan boshqariladi. Hech bir skript bu parametrlarni
hardcode qilmaydi.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Asosiy yo'llar
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"          # dataset ildizi (data.yaml shu yerda)
DATA_YAML = DATA_DIR / "data.yaml"    # Ultralytics dataset fayli

RUNS_DIR = BASE_DIR / "runs"          # Ultralytics training/output ildizi
RESULTS_DIR = BASE_DIR / "results"    # yakuniy dissertatsiya natijalari
GRAPHS_DIR = RESULTS_DIR / "graphs"

# ─────────────────────────────────────────────────────────────
# Classlar (kovsh-motion.v6i.coco datasetidan)
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "kovsh_motion",
    "bucket_digging",
    "bucket_empty",
    "bucket_loaded",
    "dumping",
    "tooth",
    "truck_empty",
    "truck_loading",
    "tuck",
]
NC = len(CLASS_NAMES)

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
# "yolo11m.pt"  -> internetdan avtomatik yuklanadi.
# Boshqa model: root/models/ nomida fayl qo'ysangiz (`best.pt`,
# `exp-2.pt`, ...) shu yo'lni ko'rsatish kifoya.
MODEL_NAME = "yolo11m.pt"

# ─────────────────────────────────────────────────────────────
# Training parametrlari
# ─────────────────────────────────────────────────────────────
TRAIN_EPOCHS = 100
IMG_SIZE = 640
BATCH = 4              # RTX 2050 (4GB) uchun; OOM bo'lsa 2 ga tushiring
DEVICE = 0
WORKERS = 4
PATIENCE = 20
PRETRAINED = True
SEED = 42

PROJECT_NAME = "kovsh_motion_exp1"     # runs/<PROJECT_NAME>/  da saqlanadi
EXP_DIR = RUNS_DIR / PROJECT_NAME      # eng yaxshi/oxirgi vazn shu yerda

# Training uchun qo'shimcha overrides (stabilik uchun):
# Kichik/noto'g'ri-label datasetda defolt lr0=0.01 loss portlashiga olib kelishi mumkin.
TRAIN_KWARGS = {
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
}

# ─────────────────────────────────────────────────────────────
# Datasetni video'dan yaratish parametrlari (prepare_dataset.py)
# ─────────────────────────────────────────────────────────────
SOURCE_MODEL = BASE_DIR / "models" / "exp-2.pt"  # avto-labeling modeli
SOURCE_VIDEO = BASE_DIR / "static" / "video.mp4"  # manba video
LABEL_CONF = 0.25          # avto-label confidence thresholdi
LABEL_IMGSZ = 320          # avto-label skripti uchun imgsz
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

# ─────────────────────────────────────────────────────────────
# Testing / video inference parametrlari
# ─────────────────────────────────────────────────────────────
INFERENCE_CONF = 0.25
INFERENCE_IOU = 0.45

# Mavjud mahalliy modellar ro'yxati (agar MODEL_NAME yuklanmasa)
AVAILABLE_MODELS = [
    str(p) for p in sorted({*list((BASE_DIR / "models").glob("*.pt")),
                            *list((BASE_DIR / "static").glob("*.pt"))})
]


def resolve_model() -> str:
    """MODEL_NAME qiymatini ishlatiladigan modelga aylantiradi.

    - Agar MODEL_NAME fayl bo'lsa -> o'sha model.
    - Aks holda 'yolo11m.pt' (internetdan yuklanadi).
    """
    model = Path(MODEL_NAME)
    if model.exists():
        return str(model.resolve())
    if (BASE_DIR / "models" / MODEL_NAME).exists():
        return str(BASE_DIR / "models" / MODEL_NAME)
    return MODEL_NAME


def ensure_dirs() -> None:
    """Natija papkalarini yaratadi."""
    for d in (DATA_DIR, RUNS_DIR, RESULTS_DIR, GRAPHS_DIR,
              RESULTS_DIR / "training", RESULTS_DIR / "validation",
              RESULTS_DIR / "test" / "images", RESULTS_DIR / "predictions"):
        d.mkdir(parents=True, exist_ok=True)