#!/usr/bin/env python3
"""
Convert the dataset of connection-pair / routed image pairs into a HuggingFace
dataset and upload it, ready for fine-tuning Flux 2 Klein 4B.

Reads PNG images from ./dataset/images/, excludes failures listed in
failures.json, resizes to 256x256, splits into train/test, and pushes
to HuggingFace as a standard Parquet-backed dataset.

The uploaded dataset has columns matching what the DreamBooth Klein
img2img training script expects:
    - cond_image:    input connection-pairs image (PIL Image)
    - output_image:  target routed image (PIL Image)
    - instruction:   edit instruction text (string)

Usage:
    python scripts/convert_and_upload.py
"""

import json
import os
import random

from datasets import Dataset, DatasetDict, Image
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
FAILURES_PATH = os.path.join(DATASET_DIR, "failures.json")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

EDIT_INSTRUCTION = (
    "Route the traces between the color matched pins, using red for the top "
    "layer and blue for the bottom layer.  Add vias to keep traces of the same "
    "color from crossing."
)

HF_REPO_ID = "tscircuit/zero-obstacle-high-density-z01"
TARGET_SIZE = (256, 256)
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_failure_ids(path: str) -> set[str]:
    """Return the set of problemIds listed in failures.json."""
    with open(path, "r") as f:
        failures = json.load(f)
    return {entry["problemId"] for entry in failures}


def discover_samples(images_dir: str, failure_ids: set[str]) -> list[str]:
    """Return sorted list of sample IDs that are NOT failures and have both
    connection-pairs and routed PNGs."""
    cp_dir = os.path.join(images_dir, "connection-pairs")
    routed_dir = os.path.join(images_dir, "routed")

    sample_ids = []
    for fname in sorted(os.listdir(cp_dir)):
        if not fname.endswith(".png"):
            continue
        sample_id = fname.removesuffix(".png")
        if sample_id in failure_ids:
            continue
        routed_path = os.path.join(routed_dir, fname)
        if not os.path.exists(routed_path):
            print(f"  Warning: skipping {sample_id} — missing routed image")
            continue
        sample_ids.append(sample_id)

    return sample_ids


def load_split(sample_ids: list[str], images_dir: str) -> Dataset:
    """Load images for the given sample IDs and return a HuggingFace Dataset."""
    cond_images = []
    output_images = []
    instructions = []

    for sid in sample_ids:
        cp_path = os.path.join(images_dir, "connection-pairs", f"{sid}.png")
        rt_path = os.path.join(images_dir, "routed", f"{sid}.png")

        cond_img = PILImage.open(cp_path).convert("RGB")
        out_img = PILImage.open(rt_path).convert("RGB")

        if cond_img.size != TARGET_SIZE:
            cond_img = cond_img.resize(TARGET_SIZE, PILImage.LANCZOS)
        if out_img.size != TARGET_SIZE:
            out_img = out_img.resize(TARGET_SIZE, PILImage.LANCZOS)

        cond_images.append(cond_img)
        output_images.append(out_img)
        instructions.append(EDIT_INSTRUCTION)

    ds = Dataset.from_dict(
        {
            "cond_image": cond_images,
            "output_image": output_images,
            "instruction": instructions,
        }
    )
    ds = ds.cast_column("cond_image", Image())
    ds = ds.cast_column("output_image", Image())
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1. Load failures
    failure_ids = load_failure_ids(FAILURES_PATH)
    print(f"Loaded {len(failure_ids)} unique failure IDs to exclude: {sorted(failure_ids)}")

    # 2. Discover valid samples
    sample_ids = discover_samples(IMAGES_DIR, failure_ids)
    print(f"Found {len(sample_ids)} valid samples (after excluding failures)")

    if not sample_ids:
        print("No valid samples found. Exiting.")
        return

    # 3. Shuffle and split into train / test
    rng = random.Random(RANDOM_SEED)
    shuffled = sample_ids.copy()
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - TEST_RATIO)))
    train_ids = sorted(shuffled[:split_idx])
    test_ids = sorted(shuffled[split_idx:])

    print(f"Split: {len(train_ids)} train, {len(test_ids)} test")

    # 4. Build HuggingFace datasets
    print("\nLoading images...")
    train_ds = load_split(train_ids, IMAGES_DIR)
    test_ds = load_split(test_ids, IMAGES_DIR)

    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})
    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    print(f"Columns: {train_ds.column_names}")
    print(f"Features: {train_ds.features}")

    # 5. Push to HuggingFace
    print(f"\nPushing to HuggingFace: {HF_REPO_ID} ...")
    dataset_dict.push_to_hub(HF_REPO_ID)
    print(f"  Uploaded train and test splits.")

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
    print(f"\nUsage:")
    print(f'  from datasets import load_dataset')
    print(f'  ds = load_dataset("{HF_REPO_ID}")')
    print(f'  ds["train"][0]  # {{"cond_image": <PIL>, "output_image": <PIL>, "instruction": "..."}}'
    )


if __name__ == "__main__":
    main()
