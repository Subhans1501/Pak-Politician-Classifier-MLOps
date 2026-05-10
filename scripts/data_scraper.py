"""
Robust image scraper for Pakistani politician face photos.

Uses DuckDuckGo Image Search to find face photos of 16 Pakistani politicians.
Each downloaded image is validated with:
  - PIL integrity check (not corrupted)
  - Minimum file size (5 KB) and dimension (80×80 px)
  - OpenCV face detection (must contain at least one face)
  - Content-hash deduplication (no exact duplicate images)
"""

import os
import sys
import time
import hashlib
import logging
import requests
import cv2
import numpy as np
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# ---------------------------------------------------------------------------
# Import DuckDuckGo search
# ---------------------------------------------------------------------------
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("ERROR: Please install ddgs:  pip install ddgs")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_IMAGE_BYTES  = 5_000       # reject images smaller than 5 KB
MIN_IMAGE_DIM    = 80          # reject images smaller than 80×80 px
MIN_PER_CLASS    = 80          # minimum images needed per class
MAX_PER_CLASS    = 140         # stop downloading after this many
DDG_PER_QUERY    = 60          # max results per DDG query

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# Load OpenCV face detector once (Haar cascade)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------------------------
# 16 politicians – targeted face-specific queries
# ---------------------------------------------------------------------------
POLITICIANS = {
    "imran_khan": {
        "name": "Imran Khan",
        "queries": [
            "Imran Khan face photo",
            "Imran Khan portrait close up",
            "Imran Khan PM Pakistan photo",
            "Imran Khan PTI chairman face",
            "Imran Khan press conference close up",
        ],
    },
    "nawaz_sharif": {
        "name": "Nawaz Sharif",
        "queries": [
            "Nawaz Sharif face photo",
            "Nawaz Sharif portrait close up",
            "Nawaz Sharif PM Pakistan photo",
            "Nawaz Sharif PML-N leader face",
            "Nawaz Sharif press conference close up",
        ],
    },
    "shehbaz_sharif": {
        "name": "Shehbaz Sharif",
        "queries": [
            "Shehbaz Sharif face photo",
            "Shahbaz Sharif portrait close up",
            "Shehbaz Sharif PM Pakistan photo",
            "Shahbaz Sharif PML-N leader face",
            "Shehbaz Sharif press conference close up",
        ],
    },
    "asif_ali_zardari": {
        "name": "Asif Ali Zardari",
        "queries": [
            "Asif Ali Zardari face photo",
            "Asif Zardari portrait close up",
            "Asif Ali Zardari President Pakistan photo",
            "Asif Zardari PPP politician face",
            "Asif Ali Zardari press conference close up",
        ],
    },
    "maryam_nawaz": {
        "name": "Maryam Nawaz",
        "queries": [
            "Maryam Nawaz face photo",
            "Maryam Nawaz portrait close up",
            "Maryam Nawaz CM Punjab photo",
            "Maryam Nawaz PML-N leader face",
            "Maryam Nawaz Sharif press conference close up",
        ],
    },
    "bilawal_bhutto": {
        "name": "Bilawal Bhutto Zardari",
        "queries": [
            "Bilawal Bhutto Zardari face photo",
            "Bilawal Bhutto portrait close up",
            "Bilawal Bhutto PPP chairman photo",
            "Bilawal Bhutto foreign minister face",
            "Bilawal Bhutto press conference close up",
        ],
    },
    "maulana_fazlur_rehman": {
        "name": "Maulana Fazlur Rehman",
        "queries": [
            "Maulana Fazlur Rehman face photo",
            "Fazlur Rehman portrait close up",
            "Maulana Fazlur Rehman JUI-F photo",
            "Fazlur Rehman Pakistan politician face",
            "Maulana Fazlur Rehman press conference close up",
        ],
    },
    "pervez_musharraf": {
        "name": "Pervez Musharraf",
        "queries": [
            "Pervez Musharraf face photo",
            "Pervez Musharraf portrait close up",
            "Pervez Musharraf President Pakistan photo",
            "General Musharraf face",
            "Pervez Musharraf press conference close up",
        ],
    },
    "benazir_bhutto": {
        "name": "Benazir Bhutto",
        "queries": [
            "Benazir Bhutto face photo",
            "Benazir Bhutto portrait close up",
            "Benazir Bhutto PM Pakistan photo",
            "Benazir Bhutto PPP leader face",
            "Benazir Bhutto press conference close up",
        ],
    },
    "fawad_chaudhry": {
        "name": "Fawad Chaudhry",
        "queries": [
            "Fawad Chaudhry face photo",
            "Fawad Chaudhry portrait close up",
            "Fawad Chaudhry PTI politician photo",
            "Fawad Chaudhary information minister face",
            "Fawad Chaudhry press conference close up",
        ],
    },
    "shireen_mazari": {
        "name": "Shireen Mazari",
        "queries": [
            "Shireen Mazari face photo",
            "Dr Shireen Mazari portrait close up",
            "Shireen Mazari PTI politician photo",
            "Shireen Mazari human rights minister face",
            "Shireen Mazari press conference close up",
        ],
    },
    "jahangir_tareen": {
        "name": "Jahangir Tareen",
        "queries": [
            "Jahangir Tareen face photo",
            "Jahangir Khan Tareen portrait close up",
            "Jahangir Tareen PTI politician photo",
            "Jahangir Tareen businessman politician face",
            "Jahangir Tareen press conference close up",
        ],
    },
    "hamza_shahbaz": {
        "name": "Hamza Shahbaz",
        "queries": [
            "Hamza Shahbaz Sharif face photo",
            "Hamza Shehbaz portrait close up",
            "Hamza Shahbaz PML-N politician photo",
            "Hamza Shehbaz CM Punjab face",
            "Hamza Shahbaz press conference close up",
        ],
    },
    "murad_ali_shah": {
        "name": "Murad Ali Shah",
        "queries": [
            "Murad Ali Shah face photo",
            "Murad Ali Shah portrait close up",
            "Murad Ali Shah CM Sindh photo",
            "Murad Ali Shah PPP politician face",
            "Murad Ali Shah press conference close up",
        ],
    },
    "ali_wazir": {
        "name": "Ali Wazir",
        "queries": [
            "Ali Wazir face photo",
            "Ali Wazir portrait close up",
            "Ali Wazir MNA Pakistan photo",
            "Ali Wazir PTM politician face",
            "Ali Wazir parliament press conference",
        ],
    },
    "lt_gen_ahmed_sharif_chaudhry": {
        "name": "Lt Gen Ahmed Sharif Chaudhry",
        "queries": [
            "DG ISPR Ahmed Sharif Chaudhry face photo",
            "Lt Gen Ahmed Sharif portrait close up",
            "Ahmed Sharif Chaudhry Pakistan Army photo",
            "DG ISPR Pakistan Army spokesman face",
            "Ahmed Sharif Chaudhry press conference close up",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Image Validation (with face detection)
# ═══════════════════════════════════════════════════════════════════════════

def has_face(filepath: str) -> bool:
    """Check if an image contains at least one detectable face."""
    try:
        img = cv2.imread(filepath)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        return len(faces) > 0
    except Exception:
        return False


def is_valid_image(filepath: str) -> bool:
    """Check if a file is a valid image with correct size."""
    try:
        if os.path.getsize(filepath) < MIN_IMAGE_BYTES:
            return False
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            w, h = img.size
            if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                return False
        return True
    except Exception:
        return False


def is_valid_face_image(filepath: str) -> bool:
    """Full validation: valid image + contains a face."""
    return is_valid_image(filepath) and has_face(filepath)


def count_images(directory: str) -> int:
    """Count image files in a directory."""
    if not os.path.isdir(directory):
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    return sum(
        1 for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )


def get_content_hashes(directory: str) -> set:
    """Get MD5 content hashes of all files in directory for dedup."""
    hashes = set()
    if not os.path.isdir(directory):
        return hashes
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp):
            try:
                with open(fp, "rb") as fh:
                    hashes.add(hashlib.md5(fh.read()).hexdigest())
            except Exception:
                pass
    return hashes


def clean_directory(directory: str) -> int:
    """Remove invalid/broken/faceless images. Returns count removed."""
    if not os.path.isdir(directory):
        return 0
    removed = 0
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp) and not is_valid_image(fp):
            os.remove(fp)
            removed += 1
    return removed


# ═══════════════════════════════════════════════════════════════════════════
# Download + Validate (with face detection & dedup)
# ═══════════════════════════════════════════════════════════════════════════

def download_and_validate(url: str, save_path: str,
                          existing_hashes: set) -> bool:
    """Download image, validate format/size, check for face, deduplicate."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return False
        data = resp.content
        if len(data) < MIN_IMAGE_BYTES:
            return False

        # Content-hash dedup: skip if identical image already exists
        content_hash = hashlib.md5(data).hexdigest()
        if content_hash in existing_hashes:
            return False

        # Detect extension from magic bytes
        if data[:2] == b"\xff\xd8":
            ext = "jpg"
        elif data[:4] == b"\x89PNG":
            ext = "png"
        elif data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WEBP":
            ext = "webp"
        elif data[:3] == b"GIF":
            ext = "gif"
        else:
            ext = "jpg"

        final_path = f"{save_path}.{ext}"
        with open(final_path, "wb") as f:
            f.write(data)

        # Validate: correct format + size
        if not is_valid_image(final_path):
            os.remove(final_path)
            return False

        # Face detection: must contain at least one face
        if not has_face(final_path):
            os.remove(final_path)
            return False

        # Record hash so future downloads in same batch skip this
        existing_hashes.add(content_hash)
        return True

    except Exception:
        # Clean up partial file if it exists
        for ext in ["jpg", "png", "webp", "gif"]:
            p = f"{save_path}.{ext}"
            if os.path.exists(p):
                os.remove(p)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# DuckDuckGo Image Search
# ═══════════════════════════════════════════════════════════════════════════

def get_ddg_urls(query: str, max_results: int = DDG_PER_QUERY) -> list:
    """Get image URLs from DuckDuckGo."""
    try:
        ddgs = DDGS()
        results = ddgs.images(query, max_results=max_results)
        return [r.get("image", "") for r in results if r.get("image")]
    except Exception as e:
        print(f"    DDG error: {e}")
        return []


def download_batch(urls: list, save_dir: str, prefix: str,
                   existing_hashes: set, max_total: int) -> int:
    """Download a batch of URLs with face validation. Stops at max_total."""
    os.makedirs(save_dir, exist_ok=True)
    downloaded = 0
    idx = count_images(save_dir)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for url in urls:
            # Check if we already have enough
            current = count_images(save_dir)
            if current >= max_total:
                break

            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            save_path = os.path.join(save_dir, f"{prefix}_{idx:04d}_{url_hash}")
            idx += 1
            future = executor.submit(
                download_and_validate, url, save_path, existing_hashes
            )
            futures[future] = url

        for future in as_completed(futures):
            if future.result():
                downloaded += 1

    return downloaded


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def collect_data(dataset_dir: str = "dataset/raw",
                 min_target: int = MIN_PER_CLASS,
                 max_target: int = MAX_PER_CLASS):
    """
    Main data collection pipeline.

    For each politician:
      1. Clean existing broken images
      2. Run DuckDuckGo image search with multiple query variants
      3. Download, validate (face detection + dedup), and save
      4. Stop at max_target (140) images
      5. Warn if below min_target (80) images
    """
    print("=" * 65)
    print("  Pakistani Politician Face Image Scraper")
    print("  Engine: DuckDuckGo | Validation: OpenCV Face Detection")
    print("=" * 65)
    print(f"  Min images  : {min_target} per class")
    print(f"  Max images  : {max_target} per class")
    print(f"  Classes     : {len(POLITICIANS)}")
    print(f"  Face detect : Haar Cascade (frontal face)")
    print(f"  Output      : {os.path.abspath(dataset_dir)}")
    print("=" * 65)
    print()

    os.makedirs(dataset_dir, exist_ok=True)

    for idx, (folder, info) in enumerate(POLITICIANS.items(), 1):
        save_dir = os.path.join(dataset_dir, folder)
        os.makedirs(save_dir, exist_ok=True)

        # Step 0: clean broken images
        removed = clean_directory(save_dir)
        if removed:
            print(f"  (cleaned {removed} broken images)")

        current = count_images(save_dir)
        if current >= max_target:
            print(f"[{idx:02d}/{len(POLITICIANS)}] {info['name']}: "
                  f"{current} images  [OK — at max]")
            continue

        needed = max_target - current
        print(f"[{idx:02d}/{len(POLITICIANS)}] {info['name']}: "
              f"{current} images, collecting up to {needed} more ...")

        # Build content hash set for deduplication
        existing_hashes = get_content_hashes(save_dir)

        # Run queries
        for q_idx, query in enumerate(info["queries"]):
            current = count_images(save_dir)
            if current >= max_target:
                break

            print(f"  Q{q_idx+1}/{len(info['queries'])}: "
                  f"\"{query}\" ", end="", flush=True)

            urls = get_ddg_urls(query, DDG_PER_QUERY)
            print(f"-> {len(urls)} results ", end="", flush=True)

            if urls:
                got = download_batch(
                    urls, save_dir, f"q{q_idx}",
                    existing_hashes, max_target
                )
                current = count_images(save_dir)
                print(f"-> +{got} faces (total: {current})")
            else:
                print("-> 0")

            time.sleep(1.5)

        # Final count
        final = count_images(save_dir)
        if final >= min_target:
            status = "OK"
        else:
            status = f"LOW ({final}<{min_target})"
        print(f"  => Final: {final} images  [{status}]\n")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)

    grand_total = 0
    low_classes = []

    for folder in sorted(POLITICIANS.keys()):
        fp = os.path.join(dataset_dir, folder)
        cnt = count_images(fp)
        grand_total += cnt
        if cnt >= min_target:
            status = "OK"
        else:
            status = "LOW"
            low_classes.append(folder)
        print(f"  {folder:40s}  {cnt:4d}  [{status}]")

    print(f"\n  Grand Total : {grand_total}")
    print(f"  Range       : {min_target}-{max_target} per class")
    print(f"  Classes OK  : {len(POLITICIANS) - len(low_classes)}/{len(POLITICIANS)}")

    if low_classes:
        print(f"\n  ⚠ Classes below {min_target} images:")
        for c in low_classes:
            print(f"     - {c} ({count_images(os.path.join(dataset_dir, c))})")
        print("     Run the script again or add images manually.")

    print("=" * 65)


if __name__ == "__main__":
    collect_data()
