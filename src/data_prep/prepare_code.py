"""Prepare the CODE Oral Mucosa dataset for the Set 1 pretraining pool.

Student A - step 6.

What it does:
  1. Finds the dataset versions (Version_1 ... Version_5). Each version
     contains all the previous ones, so only the LATEST version is kept.
  2. Keeps only original photos, i.e. photos inside an "UNANNOTATED" folder.
     The folder name is spelled in many ways (UNANNOTATED, Unnnotated,
     UNANNODATED, "UN ANNOTATED"...), but it always STARTS WITH "un".
  3. Skips annotated photos, JSON files, the metadata spreadsheet and
     hidden macOS files.
  4. Resizes each photo (longest side 512 px) and saves it to
     data/pool/code_oral_mucosa/
  5. Writes data/metadata/code_oral_mucosa_images.csv
     (new file name, participant ID, version, original path).

Expected result: about 5,566 photos from 704 participants.

Usage (from the repository root):
    python src/data_prep/prepare_code.py
    python src/data_prep/prepare_code.py --source "path/to/CODE_folder_or_zips"
"""

import argparse
import re
from collections import Counter, defaultdict

import pandas as pd

from common import (METADATA_DIR, POOL_DIR, RAW_DIR, is_hidden, is_image,
                    list_files, normalise, print_reasons, reset_folder,
                    resize_and_save)

DATASET = "code_oral_mucosa"
# Words that mean "this is not an original photo"
SKIP_WORDS = ["mask", "overlay", "label", "augment", "seg", "crop", "region"]


def check_file(name):
    """Return (True, "") to keep the file, or (False, reason) to skip it."""
    if not is_image(name):
        return False, "not an image (JSON, spreadsheet...)"
    if is_hidden(name):
        return False, "hidden macOS file"

    parts = [p.strip() for p in normalise(name).split("/")]
    folders = parts[:-1]

    # Rule 1: one of the folders must start with "un" (unannotated)
    if not any(f.startswith("un") for f in folders):
        return False, "not in an UNANNOTATED folder"
    # Rule 2: no other folder may be an annotated folder
    for f in folders:
        if not f.startswith("un") and "ann" in f:
            return False, f"annotated folder ({f})"
    # Rule 3: no forbidden word (masks, crops...)
    rest = "/".join(p for p in parts if not p.startswith("un"))
    for word in SKIP_WORDS:
        if word in rest:
            return False, f"contains the word '{word}'"
    return True, ""


def find_version(name):
    """'..._Version_5/...' -> 5"""
    match = re.search(r"version[_ ]?(\d+)", name.lower())
    return int(match.group(1)) if match else None


def find_participant(name):
    """'SMITA03453_KSR_DT.jpg' -> 'SMITA_3453' (same rule as the split script)."""
    match = re.search(r"smita\s*0*(\d+)", name.lower())
    return f"SMITA_{int(match.group(1))}" if match else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=RAW_DIR / "CODE_oral_mucosa",
                        help="CODE folder (extracted) or folder with the 5 zip files")
    parser.add_argument("--out", default=POOL_DIR / DATASET)
    args = parser.parse_args()

    files = list_files(args.source)
    print(f"{len(files)} files found in {args.source}")

    versions = sorted({v for v in (find_version(n) for n, _ in files) if v is not None})
    latest = versions[-1] if versions else None
    if versions:
        print(f"Versions found: {versions} -> keeping only Version_{latest}")

    reset_folder(args.out)
    rows, reasons, examples = [], Counter(), defaultdict(list)

    for i, (name, read) in enumerate(files, start=1):
        if i % 2000 == 0:
            print(f"   ... {i}/{len(files)}")
        keep, reason = check_file(name)
        version = find_version(name)
        if keep and latest is not None and version != latest:
            keep, reason = False, f"older version (Version_{version})"
        if keep:
            new_name = f"{DATASET}_{len(rows) + 1:05d}.jpg"
            status = resize_and_save(read(), args.out / new_name)
            if status == "ok":
                rows.append({"file": new_name, "participant": find_participant(name),
                             "version": version, "original_path": name})
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
    print(f"Photos kept   : {len(rows)}  (expected: about 5566)")
    if not table.empty:
        print(f"Participants  : {table['participant'].replace('', pd.NA).nunique()}  (expected: about 704)")
    print(f"Files skipped : {sum(reasons.values())}")
    print_reasons(reasons, examples)
    print(f"\nPhotos saved to : {args.out}")
    print(f"List saved to   : {list_path}")


if __name__ == "__main__":
    main()