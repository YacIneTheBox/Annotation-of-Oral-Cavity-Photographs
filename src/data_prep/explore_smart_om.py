"""Explore the SMART-OM dataset (evaluation set, Set 2).

Student A - step 2. This script only reads files, it changes nothing.

What it prints:
  1. The number of photos per class (unannotated photos only).
  2. The content of the Metadata and Descriptors folders.
  3. Where the patient ID is written.

Usage (from the repository root):
    python src/data_prep/explore_smart_om.py
    python src/data_prep/explore_smart_om.py --smart-om "path/to/SMART-OM"
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import IMAGE_EXTENSIONS, RAW_DIR

CLASSES = {
    "01. Normal": "normal",
    "02. Variation from normal": "normal_variation",
    "03. OPMD": "opmd",
    "04. Oral Cancer": "cancer",
}


def show_json_structure(data, depth=0, max_depth=3):
    """Print the keys of a JSON file to see how it is organised."""
    pad = "    " * depth
    if depth >= max_depth:
        return
    if isinstance(data, dict):
        for i, (key, value) in enumerate(data.items()):
            if i >= 15:
                print(f"{pad}... ({len(data)} keys in total)")
                break
            preview = value if isinstance(value, (str, int, float)) else type(value).__name__
            print(f"{pad}- {key}: {preview}")
            show_json_structure(value, depth + 1, max_depth)
    elif isinstance(data, list) and data:
        print(f"{pad}[list of {len(data)} items, first one:]")
        show_json_structure(data[0], depth + 1, max_depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smart-om", default=RAW_DIR / "SMART-OM")
    args = parser.parse_args()
    root = Path(args.smart_om)

    # The class folders may be one level deeper (SMART-OM/SMART-OM/01. Normal)
    first_class = next(root.rglob("01. Normal"), None)
    if first_class is None:
        raise SystemExit(f"'01. Normal' folder not found under {root}")
    root = first_class.parent
    print(f"SMART-OM root: {root}\n")

    print("=" * 60)
    print("1. PHOTOS PER CLASS (unannotated only)")
    print("=" * 60)
    total, patients = 0, set()
    for folder, label in CLASSES.items():
        photos = [f for f in (root / folder).rglob("*")
                  if f.suffix.lower() in IMAGE_EXTENSIONS and "unannotated" in str(f).lower()
                  and not f.name.startswith(".")]
        total += len(photos)
        for f in photos:
            m = re.match(r"^smita0*(\d+)", f.stem.lower())
            patients.add(f"SMITA_{int(m.group(1))}" if m else f.stem)
        print(f"{label:18s}: {len(photos):5d} photos   e.g. {photos[0].name if photos else '-'}")
    print(f"{'TOTAL':18s}: {total:5d} photos (expected: 2469)")
    print(f"Distinct patient IDs read from file names: {len(patients)} (paper: 331 subjects)")

    for name in ["Metadata", "Descriptors"]:
        folder = next(root.parent.rglob(name), None) or root / name
        print("\n" + "=" * 60)
        print(f"2. CONTENT OF {name.upper()}")
        print("=" * 60)
        if not folder.exists():
            print("Folder not found.")
            continue
        files = [f for f in folder.rglob("*") if f.is_file()]
        print(f"{len(files)} files. Extensions: {dict(Counter(f.suffix.lower() for f in files))}")
        json_files = [f for f in files if f.suffix.lower() == ".json"]
        if json_files:
            print(f"\nStructure of {json_files[0].name}:")
            try:
                show_json_structure(json.loads(json_files[0].read_text(encoding="utf-8")))
            except Exception as error:
                print(f"Could not read it: {error}")


if __name__ == "__main__":
    main()