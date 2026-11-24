"""Prepare the PhotoMOCI dataset for the Set 1 pretraining pool.

Student A - step 4.

What it does:
  1. Keeps only the photos inside the "oral1" folder.
  2. Skips the COCO annotation files (train/val/test/dataset JSON)
     and the CBR-case-base.csv file.
  3. Resizes each photo (longest side 512 px) and saves it to
     data/pool/photomoci/
  4. Writes data/metadata/photomoci_images.csv (new name -> original path).

Expected result: about 700 photos.

Usage (from the repository root):
    python src/data_prep/prepare_photomoci.py
    python src/data_prep/prepare_photomoci.py --source "path/to/archive.zip"
"""

import argparse
from collections import Counter, defaultdict

import pandas as pd

from common import (METADATA_DIR, POOL_DIR, RAW_DIR, is_hidden, is_image,
                    list_files, normalise, print_reasons, reset_folder,
                    resize_and_save)

DATASET = "photomoci"
PHOTO_FOLDER = "oral1"  # the folder that holds the photos in PhotoMOCI


def check_file(name):
    """Return (True, "") to keep the file, or (False, reason) to skip it."""
    if not is_image(name):
        return False, "not an image (JSON, CSV...)"
    if is_hidden(name):
        return False, "hidden macOS file"
    if PHOTO_FOLDER not in normalise(name).split("/"):
        return False, f"not in the '{PHOTO_FOLDER}' folder"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=RAW_DIR / "PhotoMOCI",
                        help="PhotoMOCI folder (extracted) or archive.zip")
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
                rows.append({"file": new_name, "original_path": name})
                continue
            reason = f"image {status}"
        reasons[reason] += 1
        if len(examples[reason]) < 3:
            examples[reason].append(name)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    list_path = METADATA_DIR / f"{DATASET}_images.csv"
    pd.DataFrame(rows).to_csv(list_path, index=False)

    print("\n" + "=" * 60)
    print(f"Photos kept   : {len(rows)}  (expected: about 700)")
    print(f"Files skipped : {sum(reasons.values())}")
    print_reasons(reasons, examples)
    print(f"\nPhotos saved to : {args.out}")
    print(f"List saved to   : {list_path}")


if __name__ == "__main__":
    main()