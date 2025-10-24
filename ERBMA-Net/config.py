import os

# === Base Directories (Change these before running) ===
BASE_DATASET_DIR = os.path.join("data", "AVEC2014")
OUTPUT_DIR = os.path.join("results")

# === CSV Output Paths ===
OUTPUT_CSV_FILES = {
    'train': os.path.join(OUTPUT_DIR, 'train_features.csv'),
    'dev': os.path.join(OUTPUT_DIR, 'dev_features.csv'),
    'test': os.path.join(OUTPUT_DIR, 'test_features.csv'),
}

# === Augmentation Settings ===
AUGMENT = True        # Whether to apply augmentation
AUGMENT_COUNT = 10    # Number of augmentations per frame
