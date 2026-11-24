"""Shared helpers for the data preparation scripts (Student A).

Every prepare_*.py script uses these functions to:
  - read files from a dataset folder or zip archive,
  - resize photos and save them as JPEG,
  - keep the default folder layout of the repository in one place.
"""

import io
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Repository layout (paths are relative to the repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"            # downloaded datasets (not on git)
POOL_DIR = REPO_ROOT / "data" / "pool"          # cleaned Set 1 photos (not on git)
METADATA_DIR = REPO_ROOT / "data" / "metadata"  # small CSV lists (on git)

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_SIZE = 512   # longest side in pixels after resizing (DINO uses 224)
MIN_SIZE = 100   # images smaller than this are skipped
JPEG_QUALITY = 90


def normalise(path_text):
    """Lower-case path with forward slashes, for easy matching."""
    return str(path_text).replace("\\", "/").lower()


def is_image(path_text):
    return Path(normalise(path_text)).suffix in IMAGE_EXTENSIONS


def is_hidden(path_text):
    """Hidden or macOS helper files (__MACOSX folders, ._ files)."""
    text = "/" + normalise(path_text)
    return "/__macosx/" in text or Path(text).name.startswith(".")


def list_files(source):
    """List every file of a dataset as (name, read_function) pairs.

    `source` can be:
      - a .zip file,
      - a folder that contains .zip files (read inside them, no extraction),
      - a normal folder with the files already extracted.
    `name` is the path inside the dataset, with forward slashes.
    """
    source = Path(source)
    files = []

    def add_zip(zip_path):
        archive = zipfile.ZipFile(zip_path)
        for name in archive.namelist():
            if not name.endswith("/"):
                files.append((name, lambda n=name, a=archive: a.read(n)))

    if source.is_file() and source.suffix.lower() == ".zip":
        add_zip(source)
    elif source.is_dir():
        zips = sorted(source.rglob("*.zip"))
        if zips:
            for z in zips:
                add_zip(z)
        else:
            for f in sorted(source.rglob("*")):
                if f.is_file():
                    name = f.relative_to(source).as_posix()
                    files.append((name, lambda p=f: p.read_bytes()))
    else:
        raise FileNotFoundError(f"Dataset not found: {source}")
    return files


def resize_and_save(image_bytes, output_path):
    """Open a photo, fix its rotation, resize it and save it as JPEG.

    Returns "ok", "too small" or "broken".
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)  # phone photos are often stored rotated
        image = image.convert("RGB")
    except Exception:
        return "broken"
    if min(image.size) < MIN_SIZE:
        return "too small"
    image.thumbnail((MAX_SIZE, MAX_SIZE))  # keeps the aspect ratio
    image.save(output_path, "JPEG", quality=JPEG_QUALITY)
    return "ok"


def reset_folder(folder):
    """Empty an output folder so old and new results are never mixed."""
    folder = Path(folder)
    if folder.exists():
        print(f"Emptying previous output folder: {folder}")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def print_reasons(reasons, examples):
    """Print how many files were skipped for each reason, with examples."""
    if not reasons:
        return
    print("\nWHY files were skipped:")
    for reason, count in reasons.most_common():
        print(f"  {count:6d}  {reason}")
        for example in examples[reason]:
            print(f"            e.g. {example[-90:]}")