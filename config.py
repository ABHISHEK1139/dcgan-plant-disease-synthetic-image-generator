"""Central configuration for the DCGAN plant-disease image generator.

Paths and hyperparameters shared by every module under ``src/``.
Values mirror the training spec documented in README.md.
"""

import os

# ── Data ──────────────────────────────────────────────────────────
# Root of the PlantVillage-style dataset: <DATA_DIR>/<Plant>___<Disease>/*.jpg
DATA_DIR = "data"
IMG_SIZE = 128          # README: 128x128 pixels
IMG_CHANNELS = 3        # README: RGB
NUM_WORKERS = 4         # DataLoader workers

# ── Model ─────────────────────────────────────────────────────────
LATENT_DIM = 100        # README: latent dimension 100

# ── Training ──────────────────────────────────────────────────────
BATCH_SIZE = 64         # README: batch size 64
STAGE1_EPOCHS = 200     # README: Stage 1 (anatomy) epochs
STAGE2_EPOCHS = 200     # README: Stage 2 (disease) epochs
STAGE1_LR = 0.0002      # README: Stage 1 learning rate
STAGE2_LR = 1e-5        # README: Stage 2 generator LR (TTUR)
STAGE2_D_LR = 2e-4      # README: Stage 2 discriminator LR (TTUR)
BETA1 = 0.5             # README: Adam Beta1
BETA2 = 0.999           # README: Adam Beta2

# ── Stabilization ─────────────────────────────────────────────────
LABEL_SMOOTHING_REAL = 0.9   # README: real label smoothing
LABEL_SMOOTHING_FAKE = 0.0   # README: fake label smoothing
LABEL_FLIP_RATE = 0.05       # 5% label flipping for D
NOISE_STD_INITIAL = 0.1      # initial Gaussian noise std on D inputs
NOISE_DECAY = 0.99           # per-epoch decay of the noise std
GRAD_CLIP_VALUE = 1.0        # gradient clipping norm

# ── Output ────────────────────────────────────────────────────────
SAMPLE_INTERVAL = 5     # save sample grids every N epochs
CHECKPOINT_DIR = "checkpoints"
SAMPLES_DIR = "samples"
LOGS_DIR = "logs"

for _d in (CHECKPOINT_DIR, SAMPLES_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)
del _d

# ── Dataset classes (PlantVillage) ────────────────────────────────
PLANT_DISEASE_MAP = {
    "Apple": ["Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"],
    "Blueberry": ["healthy"],
    "Cherry": ["Powdery_mildew", "healthy"],
    "Corn": ["Cercospora_leaf_spot", "Gray_leaf_spot", "Common_rust",
             "Northern_Leaf_Blight", "healthy"],
    "Grape": ["Black_rot", "Esca", "Leaf_blight", "healthy"],
    "Orange": ["Haunglongbing"],
    "Peach": ["Bacterial_spot", "healthy"],
    "Pepper": ["Bacterial_spot", "healthy"],
    "Potato": ["Early_blight", "Late_blight", "healthy"],
    "Raspberry": ["healthy"],
    "Soybean": ["healthy"],
    "Squash": ["Powdery_mildew"],
    "Strawberry": ["Leaf_scorch", "healthy"],
    "Tomato": ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
               "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
               "Yellow_Leaf_Curl_Virus", "mosaic_virus", "healthy"],
}


def get_class_folder(plant: str, disease: str) -> str:
    """Return the dataset folder name for a plant-disease combination.

    Follows the PlantVillage convention: ``<Plant>___<Disease>``
    (e.g. ``Apple___Apple_scab``).
    """
    return f"{plant}___{disease}"
