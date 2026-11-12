"""Verify the archived submission, or regenerate it from the frozen model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from filament.frozen import calibrate_photo, validate_csv

CONFIDENCE, PIXEL = 0.90, 0.40
VIEWS = ("identity", "hflip", "vflip")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def official_inventory():
    names = (ROOT / "evidence/official-test-images.txt").read_text().splitlines()
    if len(names) != 180 or len(set(names)) != 180:
        raise ValueError("Official test inventory must contain 180 unique image names")
    return names


def validate(path, names, reference_sha):
    with path.open(newline="", encoding="utf-8") as stream:
        if csv.DictReader(stream).fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Unexpected CSV header")
    result = validate_csv(path, names)
    result["exact_archived_csv_sha256"] = result["sha256"] == reference_sha
    return result


def infer(test_images, checkpoint, output, names):
    import numpy as np
    import torch
    from PIL import Image
    from filament.hierarchical_fusion import associate, vote, mask_nms
    from filament.optimized_fusion import exclusive_soft_groups
    from filament.model import load_model
    from filament import frozen

    paths = sorted(p for p in test_images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if [p.name for p in paths] != sorted(names):
        raise ValueError("Input inventory does not match the official 180 test files")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bridge = load_model(checkpoint, ROOT / "vendor/mask2former")
    frozen.VIEWS = VIEWS
    key = (CONFIDENCE, PIXEL)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        for path in paths:
            with Image.open(path) as image:
                rgb = np.array(image.convert("L").convert("RGB"))
            predictions, _, _ = calibrate_photo(
                bridge, rgb, torch, [key], associate, vote, mask_nms, exclusive_soft_groups
            )
            for index, row in enumerate(predictions[key]):
                writer.writerow([f"{path.stem}_{index}", row["rle"]["counts"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", type=Path)
    modes.add_argument("--test-images", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/fboth-epoch80-model.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions/submission.csv")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "evidence/manifest.json").read_text())
    names = official_inventory()
    if args.test_images:
        if digest(args.checkpoint) != manifest["inference_model_sha256"]:
            raise ValueError("Inference weight SHA-256 differs from the release manifest")
        infer(args.test_images, args.checkpoint, args.output, names)
    result = validate(args.validate_only or args.output, names, manifest["submission_csv_sha256"])
    print(json.dumps(result, indent=2))
    if not result["exact_archived_csv_sha256"]:
        raise SystemExit("CSV validation passed but the SHA-256 differs from the submitted CSV")


if __name__ == "__main__":
    main()
