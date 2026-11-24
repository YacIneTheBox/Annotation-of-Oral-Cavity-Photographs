"""Pack the final Set 1 photos into one zip file, to upload to Kaggle.

Student A - step 8.1.

What it does:
  1. Reads data/metadata/pool.csv (the photos kept after deduplication).
  2. Puts only those photos into data/set1_pool.zip,
     with one folder per dataset (code_oral_mucosa/, bmiocd/, photomoci/).
  3. Adds pool.csv inside the zip, so the training notebook knows the list.

Usage (from the repository root):
    python src/data_prep/make_pool_zip.py
"""

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from common import METADATA_DIR, POOL_DIR, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=REPO_ROOT / "data" / "set1_pool.zip")
    args = parser.parse_args()

    pool = pd.read_csv(METADATA_DIR / "pool.csv")
    out = Path(args.out)
    missing = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:  # JPEGs are already compressed
        for relative in pool["file"]:
            photo = POOL_DIR / relative
            if photo.exists():
                archive.write(photo, relative)
            else:
                missing += 1
        archive.write(METADATA_DIR / "pool.csv", "pool.csv")

    print(f"{len(pool) - missing} photos written to {out} ({out.stat().st_size / 1024 ** 2:.0f} MB)")
    if missing:
        print(f"WARNING: {missing} photos listed in pool.csv were not found in {POOL_DIR}")


if __name__ == "__main__":
    main()