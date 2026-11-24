"""Remove duplicates and SMART-OM look-alikes from the Set 1 pool.

Student A - step 7.

The evaluation set (SMART-OM) must never be in the pretraining pool,
otherwise the model would have "seen the exam" before taking it.

This script checks 3 things:
  1. No CODE participant has the same SMITA ID as a SMART-OM patient.
  2. No pool photo looks like a SMART-OM photo (perceptual hash, pHash).
  3. No photo appears twice in the pool.

It does NOT delete any photo. It writes two lists:
  - data/metadata/pool.csv    : photos kept  -> the final Set 1 for DINO
  - data/metadata/removed.csv : photos removed, with the reason

Usage (from the repository root):
    python src/data_prep/deduplicate_pool.py
    python src/data_prep/deduplicate_pool.py --smart-om "path/to/SMART-OM"
"""

import argparse
import re
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from common import IMAGE_EXTENSIONS, METADATA_DIR, POOL_DIR, RAW_DIR, REPO_ROOT

THRESHOLD = 6  # hash distance: 0 = identical, 64 = completely different


def phash(path):
    """Perceptual hash of a photo, as an integer (None if unreadable).
    Two near-identical photos have near-identical hashes, even if resized."""
    try:
        with Image.open(path) as image:
            return int(str(imagehash.phash(image.convert("RGB"))), 16)
    except Exception:
        return None


def hamming(one, many):
    """Number of different bits between one hash and an array of hashes."""
    xor = np.bitwise_xor(many, np.uint64(one))
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def smita_id(text):
    """'SMITA03453_KSR_DT.jpg' -> 'SMITA_3453' (same rule as the split script)."""
    match = re.search(r"smita\s*0*(\d+)", text.lower())
    return f"SMITA_{int(match.group(1))}" if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default=POOL_DIR)
    parser.add_argument("--smart-om", default=RAW_DIR / "SMART-OM")
    parser.add_argument("--code-list", default=METADATA_DIR / "code_oral_mucosa_images.csv")
    parser.add_argument("--exam-list", default=REPO_ROOT / "splits" / "exam_list.txt",
                        help="exam photos from Student B's split (used for the report)")
    args = parser.parse_args()
    pool_dir, smart_dir = Path(args.pool), Path(args.smart_om)
    code_list, exam_list = Path(args.code_list), Path(args.exam_list)
    for folder in (pool_dir, smart_dir):
        if not folder.exists():
            raise SystemExit(f"Folder not found: {folder}")

    # 1. SMART-OM photos (only the "01. Unannotated" folders)
    print("1. Reading SMART-OM...")
    smart_files = [f for f in smart_dir.rglob("*")
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                   and not f.name.startswith(".") and "unannotated" in str(f).lower()]
    smart_names, smart_hashes = [], []
    for i, f in enumerate(smart_files, start=1):
        if i % 500 == 0:
            print(f"   ... {i}/{len(smart_files)}")
        h = phash(f)
        if h is not None:
            smart_names.append(f.name)
            smart_hashes.append(h)
    smart_hashes = np.array(smart_hashes, dtype=np.uint64)
    smart_ids = {smita_id(n) for n in smart_names} - {None}
    print(f"   {len(smart_hashes)} SMART-OM photos (expected: about 2469)")
    print(f"   {len(smart_ids)} SMITA IDs found in SMART-OM")
    if len(smart_hashes) == 0:
        raise SystemExit("   No SMART-OM photo found. Check the --smart-om path.")

    # 2. Participant IDs shared by CODE and SMART-OM
    code_ids, shared = {}, set()
    if code_list.exists():
        table = pd.read_csv(code_list).fillna("")
        code_ids = dict(zip(table["file"], table["participant"]))
        shared = (set(code_ids.values()) - {""}) & smart_ids
        print(f"\n2. SMITA IDs shared by CODE and SMART-OM: {len(shared)}")
        if shared:
            print(f"   e.g. {sorted(shared)[:10]}")
            print("   -> to be safe, photos of these participants are removed")
    else:
        print(f"\n2. {code_list.name} not found: SMITA ID check skipped")

    # 3. Compare every pool photo
    print("\n3. Checking the pool photos...")
    pool_files = sorted(f for f in pool_dir.rglob("*")
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    kept, removed, kept_hashes = [], [], []

    for i, f in enumerate(pool_files, start=1):
        if i % 500 == 0:
            print(f"   ... {i}/{len(pool_files)}")
        relative = f.relative_to(pool_dir).as_posix()  # e.g. "bmiocd/bmiocd_00001.jpg"
        dataset = relative.split("/")[0]

        def drop(reason, match=""):
            removed.append({"file": relative, "dataset": dataset,
                            "reason": reason, "matches": match})

        participant = code_ids.get(f.name, "")
        if participant and participant in shared:
            drop("same SMITA ID as SMART-OM", participant)
            continue
        h = phash(f)
        if h is None:
            drop("unreadable image")
            continue
        d = hamming(h, smart_hashes)
        j = int(d.argmin())
        if d[j] <= THRESHOLD:
            drop("looks like a SMART-OM photo", smart_names[j])
            continue
        if kept_hashes:
            d2 = hamming(h, np.array(kept_hashes, dtype=np.uint64))
            k = int(d2.argmin())
            if d2[k] <= THRESHOLD:
                drop("duplicate inside the pool", kept[k]["file"])
                continue
        kept.append({"file": relative, "dataset": dataset})
        kept_hashes.append(h)

    # 4. Save and summarise
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kept).to_csv(METADATA_DIR / "pool.csv", index=False)
    pd.DataFrame(removed, columns=["file", "dataset", "reason", "matches"]).to_csv(
        METADATA_DIR / "removed.csv", index=False)

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"Photos checked : {len(pool_files)}")
    print(f"Photos kept    : {len(kept)}")
    print(f"Photos removed : {len(removed)}")
    if kept:
        print("\nPhotos kept per dataset:")
        print(pd.DataFrame(kept)["dataset"].value_counts().to_string())
    if removed:
        print("\nPhotos removed, per reason and dataset:")
        print(pd.DataFrame(removed).groupby(["reason", "dataset"]).size().to_string())

    # 5. Link with Student B's split: were the removed look-alikes exam photos?
    if exam_list.exists():
        exam_names = {line.strip() for line in exam_list.read_text().splitlines() if line.strip()}
        lookalikes = [r["matches"] for r in removed if r["reason"] == "looks like a SMART-OM photo"]
        in_exam = sum(1 for name in lookalikes if name in exam_names)
        print(f"\nExam photos (splits/exam_list.txt): {len(exam_names)}")
        print(f"SMART-OM look-alikes removed: {len(lookalikes)}, of which {in_exam} were exam photos")
        print("-> All SMART-OM photos (train AND test) are excluded, so 0 exam photos remain in the pool.")
    else:
        print(f"\n{exam_list} not found: exam check skipped (all SMART-OM photos were still excluded)")

    print(f"\nFiles written: {METADATA_DIR / 'pool.csv'} and {METADATA_DIR / 'removed.csv'}")


if __name__ == "__main__":
    main()