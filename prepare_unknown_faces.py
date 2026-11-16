#!/usr/bin/env python3
"""Prepare disjoint unknown_val and unknown_test images from the LFW dataset.

Run from the inner face-recognition-studio project root:
    python prepare_unknown_faces.py

The script downloads the official LFW funneled archive, then selects one
identity-disjoint image per person. Each image is checked with this project's
current FaceDetector before it is copied into the unknown folders.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external_lfw"
ARCHIVE_PATH = EXTERNAL_DIR / "lfw-funneled.tgz"
EXTRACT_DIR = EXTERNAL_DIR / "extracted"
UNKNOWN_VAL = DATA_DIR / "unknown_val"
UNKNOWN_TEST = DATA_DIR / "unknown_test"
MANIFEST_PATH = DATA_DIR / "unknown_faces_manifest.csv"

# LFW official mirror and SHA-256 checksum published in scikit-learn's loader.
ARCHIVE_URL = "https://ndownloader.figshare.com/files/5976015"
ARCHIVE_SHA256 = "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"
SEED = 20261116
IMAGES_PER_SPLIT = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.is_file() and sha256_file(ARCHIVE_PATH) == ARCHIVE_SHA256:
        print(f"Đã có archive LFW hợp lệ: {ARCHIVE_PATH}")
        return

    temp_path = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".part")
    print("Đang tải LFW funneled (~200 MB). Không đóng cửa sổ terminal...")
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "Mozilla/5.0 face-recognition-dataset-preparer"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as out:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                received += len(chunk)
                if total_bytes:
                    print(f"\rĐã tải {received / total_bytes:.0%}", end="", flush=True)
        print()
        actual = sha256_file(temp_path)
        if actual != ARCHIVE_SHA256:
            raise RuntimeError(
                "SHA-256 của file tải về không khớp; đã dừng để tránh dùng dữ liệu hỏng."
            )
        temp_path.replace(ARCHIVE_PATH)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def extract_dataset() -> Path:
    if EXTRACT_DIR.exists():
        for candidate in EXTRACT_DIR.rglob("lfw_funneled"):
            if candidate.is_dir():
                return candidate

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Đang giải nén LFW...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        # Python 3.12+ data filter blocks unsafe paths and special files.
        archive.extractall(path=EXTRACT_DIR, filter="data")

    matches = [p for p in EXTRACT_DIR.rglob("lfw_funneled") if p.is_dir()]
    if not matches:
        raise RuntimeError("Không tìm thấy thư mục lfw_funneled sau khi giải nén.")
    return matches[0]


def load_project_detector():
    if not (ROOT / "faceid" / "preprocess.py").is_file():
        raise RuntimeError(
            "Hãy đặt prepare_unknown_faces.py ở thư mục gốc bên trong, "
            "nơi có thư mục faceid/ và file config.json."
        )
    sys.path.insert(0, str(ROOT))
    from faceid.preprocess import FaceDetector, load_rgb  # noqa: PLC0415
    return FaceDetector(), load_rgb


def select_images(dataset_root: Path, detector, load_rgb):
    identity_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    rng = random.Random(SEED)
    rng.shuffle(identity_dirs)

    selected = []
    for identity_dir in identity_dirs:
        image_paths = sorted(
            p for p in identity_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_paths:
            continue

        # One accepted image from each identity, so val/test identities cannot overlap.
        for image_path in image_paths:
            try:
                image = load_rgb(image_path)
                boxes = detector.detect(image)
                if not boxes:
                    continue
                # Verify that the face crop operation used by the project can handle it.
                detector.crop(image, boxes[0], margin=0.2)
                selected.append((identity_dir.name, image_path))
                break
            except (OSError, ValueError, RuntimeError, Image.DecompressionBombError):
                continue

        if len(selected) >= 2 * IMAGES_PER_SPLIT:
            break

    if len(selected) < 2 * IMAGES_PER_SPLIT:
        raise RuntimeError(
            f"Chỉ tìm được {len(selected)} ảnh đạt kiểm tra khuôn mặt; cần "
            f"{2 * IMAGES_PER_SPLIT}. Hãy kiểm tra FaceDetector hiện tại."
        )
    return selected[: 2 * IMAGES_PER_SPLIT]


def main() -> int:
    for folder in (UNKNOWN_VAL, UNKNOWN_TEST):
        if folder.exists() and any(
            p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            for p in folder.rglob("*")
        ):
            raise RuntimeError(
                f"{folder} đã có ảnh. Script không ghi đè dữ liệu hiện tại; "
                "hãy sao lưu/di chuyển ảnh cũ trước khi chạy."
            )

    download_archive()
    dataset_root = extract_dataset()
    detector, load_rgb = load_project_detector()
    selected = select_images(dataset_root, detector, load_rgb)

    UNKNOWN_VAL.mkdir(parents=True, exist_ok=True)
    UNKNOWN_TEST.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for index, (identity, source_path) in enumerate(selected):
        split = "unknown_val" if index < IMAGES_PER_SPLIT else "unknown_test"
        output_dir = UNKNOWN_VAL if split == "unknown_val" else UNKNOWN_TEST
        image_number = index + 1 if split == "unknown_val" else index - IMAGES_PER_SPLIT + 1
        filename = f"unknown_{image_number:03d}.jpg"
        destination = output_dir / filename
        shutil.copy2(source_path, destination)
        manifest_rows.append({
            "split": split,
            "filename": filename,
            "lfw_identity": identity,
            "source_file": source_path.name,
            "source_dataset": "Labeled Faces in the Wild (funneled)",
            "source_sha256": sha256_file(source_path),
        })

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nHOÀN TẤT")
    print(f"unknown_val : {IMAGES_PER_SPLIT} ảnh → {UNKNOWN_VAL}")
    print(f"unknown_test: {IMAGES_PER_SPLIT} ảnh → {UNKNOWN_TEST}")
    print(f"Manifest    : {MANIFEST_PATH}")
    print("Mỗi ảnh thuộc một danh tính LFW riêng; hai tập không dùng chung danh tính.")
    print("Lưu ý: đây là tập thử nghiệm sơ bộ từ LFW, không thay thế ảnh người thật trong điều kiện triển khai.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nLỖI: {exc}", file=sys.stderr)
        raise SystemExit(1)
