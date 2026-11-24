"""Extract the downloaded dataset archives into data/raw/.

Student A - step 3.

What it does:
  1. Finds the dataset zip files in your Downloads folder.
  2. Checks that there is enough free disk space.
  3. Extracts each dataset into its own folder:
        data/raw/CODE_oral_mucosa/   (also extracts the 5 inner zip files)
        data/raw/PhotoMOCI/
        data/raw/BMIOCD/
        data/raw/SMART-OM/
  4. Never deletes your original zip files.

Note: the prepare_*.py scripts can also read the zip files directly,
so extraction is optional if you are short on disk space.

Usage (from the repository root):
    python src/data_prep/extract_datasets.py
    python src/data_prep/extract_datasets.py --downloads "D:/my_downloads"
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from common import RAW_DIR

# How to recognise each dataset's zip file in the Downloads folder
DATASETS = {
    "CODE_oral_mucosa": "30550889*.zip",
    "PhotoMOCI": "archive*.zip",
    "BMIOCD": "Bangladesh*.zip",
    "SMART-OM": "SMART*OM*.zip",
}


def gigabytes(n_bytes):
    return n_bytes / 1024 ** 3


def extract(zip_path, folder):
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for i, member in enumerate(members, start=1):
            if "__MACOSX" in member.filename:
                continue  # useless macOS helper files
            archive.extract(member, folder)
            if i % 2000 == 0:
                print(f"      ... {i}/{len(members)} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--downloads", default=Path.home() / "Downloads",
                        help='your Downloads folder ("Téléchargements" is really "Downloads")')
    parser.add_argument("--out", default=RAW_DIR)
    args = parser.parse_args()
    downloads, out = Path(args.downloads), Path(args.out)

    found = {}
    for name, pattern in DATASETS.items():
        matches = sorted(downloads.glob(pattern))
        if matches:
            found[name] = matches[0]
            print(f"Found    {name:18s} -> {matches[0].name} ({gigabytes(matches[0].stat().st_size):.1f} GB)")
        else:
            print(f"Missing  {name:18s} -> no '{pattern}' in {downloads}")
    if not found:
        raise SystemExit("No zip file found.")

    # CODE contains zip files inside a zip file: it needs about twice its size
    needed = sum(z.stat().st_size * (2.2 if n == "CODE_oral_mucosa" else 1.2)
                 for n, z in found.items())
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    print(f"\nSpace needed: about {gigabytes(needed):.1f} GB | free: {gigabytes(free):.1f} GB")
    if free < needed:
        raise SystemExit("Not enough disk space. Run the prepare_*.py scripts on the zip files instead.")

    for name, zip_path in found.items():
        folder = out / name
        if folder.exists() and any(folder.iterdir()):
            print(f"\n{name}: already extracted, skipping.")
            continue
        print(f"\nExtracting {name}...")
        folder.mkdir(parents=True, exist_ok=True)
        extract(zip_path, folder)
        for inner in sorted(folder.rglob("*.zip")):  # CODE: the 5 inner zip files
            print(f"   Extracting inner archive: {inner.name}")
            extract(inner, inner.parent)
            inner.unlink()  # deletes only the extracted COPY, to save space
        n_images = sum(1 for f in folder.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"   Done: {n_images} images in {folder}")

    print(f"\nFinished. Datasets are in {out}")


if __name__ == "__main__":
    main()