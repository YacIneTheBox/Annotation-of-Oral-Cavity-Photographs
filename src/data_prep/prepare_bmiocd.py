"""Prepare the BMIOCD dataset (Bangladesh) for the Set 1 pretraining pool.

Student A - step 5.

What it does:
  1. Keeps the photos of the 4 hospitals (Mitford Hospital, Reshmi Dental,
     Shuvro Dental, Uposhomo Dental), from both Cancer and Non_Cancer folders.
     The labels are NOT used by DINO, but they are written in the CSV.
  2. Skips any "augmented" folder (modified copies of the same photos).
  3. Resizes each photo (longest side 512 px) and saves it to
     data/pool/bmiocd/
  4. Writes data/metadata/bmiocd_images.csv (new name, hospital, original path).

Expected result: about 800 photos.

Usage (from the repository root):
    python src/data_prep/prepare_bmiocd.py
    python src/data_prep/prepare_bmiocd.py --source "path/to/BMIOCD.zip"
"""

import argparse
from collections import Counter, defaultdict

import pandas as pd

from common import (METADATA_DIR, POOL_DIR, RAW_DIR, is_hidden, is_image,
                    list_files, normalise, print_reasons, reset_folder,
                    resize_and_save)

DATASET = "bmiocd"
HOSPITALS = ["mitford hospital", "reshmi dental", "shuvro dental", "uposhomo dental"]
SKIP_WORDS = ["augment", "annotated", "annotation", "mask", "overlay", "seg"]


def check_file(name):
    """Return (True, "") to keep the file, or (False, reason) to skip it."""
    if not is_image(name):
        return False, "not an image"
    if is_hidden(name):
        return False, "hidden macOS file"
    text = normalise(name).replace("unannotated", "")
    for word in SKIP_WORDS:
        if word in text:
            return False, f"contains the word '{word}'"
    return True, ""


def find_hospital(name):
    text = normalise(name)
    return next((h for h in HOSPITALS if h in text), "unknown")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=RAW_DIR / "BMIOCD",
                        help="BMIOCD folder (extracted) or zip file")
    parser.add_argument("--out", default=POOL_DIR / DATASET)
    args = parser.parse_args()

    files = list_files(args.source)
    print(f"{len(files)} files found in {args.source}")
    reset_folder(args.out)
    rows, reasons, examples = [], Counter(), defaultdict(list)

    for name, read in files:
        keep, reason = check_file(name)
        if keep:
            new_name = f"{DATASET}_{len(rows) + 1:05d}.jpg"
            status = resize_and_save(read(), args.out / new_name)
            if status == "ok":
                label = "cancer" if "/cancer/" in "/" + normalise(name) else "non_cancer"
                rows.append({"file": new_name, "hospital": find_hospital(name),
                             "label_not_used": label, "original_path": name})
                continue
            reason = f"image {status}"
        reasons[reason] += 1
        if len(examples[reason]) < 3:
            examples[reason].append(name)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    list_path = METADATA_DIR / f"{DATASET}_images.csv"
    table.to_csv(list_path, index=False)

    print("\n" + "=" * 60)
    print(f"Photos kept   : {len(rows)}  (expected: about 800)")
    print(f"Files skipped : {sum(reasons.values())}")
    if not table.empty:
        print("\nPhotos per hospital:")
        print(table["hospital"].value_counts().to_string())
    print_reasons(reasons, examples)
    print(f"\nPhotos saved to : {args.out}")
    print(f"List saved to   : {list_path}")


if __name__ == "__main__":
    main()