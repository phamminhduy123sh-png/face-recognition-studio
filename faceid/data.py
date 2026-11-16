"""Session-disjoint datasets for the three fixed identities.

Schema: data_dir/{dung,duy,hung}/phien_01..phien_06/image.jpg.
No files are moved, renamed or repaired by the auditor.
Stored images are treated as raw frames when crop_face=True, regardless of size.
For uniformly pre-cropped legacy data, explicitly use crop_face=False; there is
no automatic 160x160/already-cropped heuristic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from torch.utils.data import Dataset

from .preprocess import FacePreprocessor, canonical_preprocess_config, load_rgb

CLASSES = ["dung", "duy", "hung"]
SESSIONS = list(range(1, 7))
SPLIT_NAMES = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SPLITS = {"train": [1, 2, 3, 4], "val": [5], "test": [6]}


class DatasetError(ValueError):
    """An actionable dataset/config validation failure."""


def initialize_dataset(root):
    """Ensure all 18 session folders exist. Never overwrite or move user files."""
    root = Path(root)
    created = []
    for person in CLASSES:
        for session in SESSIONS:
            folder = root / person / f"phien_{session:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            created.append(str(folder))
    return created


def _session_splits(config):
    splits = config.get("splits", DEFAULT_SPLITS)
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise DatasetError("splits must define exactly train, val and test.")
    owners = {}
    for split in SPLIT_NAMES:
        sessions = splits[split]
        if not isinstance(sessions, (list, tuple)) or not sessions:
            raise DatasetError(f"splits.{split} must be a nonempty list of session numbers.")
        for session in sessions:
            if isinstance(session, bool) or not isinstance(session, int) or session not in SESSIONS:
                raise DatasetError(f"Invalid session in splits.{split}: {session}; use integers 1..6.")
            if session in owners:
                raise DatasetError(f"Session {session} is repeated in splits ({owners[session]}, {split}).")
            owners[session] = split
    if set(owners) != set(SESSIONS):
        raise DatasetError("Every session 1..6 must belong to exactly one split.")
    return owners


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_counts():
    return {
        "total": 0,
        "by_split": {split: 0 for split in SPLIT_NAMES},
        "by_person": {person: 0 for person in CLASSES},
        "by_session": {f"phien_{session:02d}": 0 for session in SESSIONS},
        "by_person_session": {person: {f"phien_{session:02d}": 0 for session in SESSIONS} for person in CLASSES},
        "by_split_person": {split: {person: 0 for person in CLASSES} for split in SPLIT_NAMES},
    }


def audit_dataset(config, strict=True):
    """Return a JSON-serializable report, including all errors; never mutate data.

    Always decodes and hashes images, checks session completeness and detects
    exact/decoded-pixel duplicates. Cross-split or cross-person duplicates are
    errors; duplicates inside the same person's same split are warnings.
    strict=True also checks face suitability and rejects unexpected directories.
    strict=False skips face detection and reports unknown directories as warnings.
    """
    root = Path(config.get("data_dir", "")).expanduser().resolve()
    report = {
        "valid": False, "root": str(root), "classes": list(CLASSES),
        "class_to_idx": {person: i for i, person in enumerate(CLASSES)},
        "counts": _empty_counts(), "manifest": [], "issues": [], "duplicates": [],
        "signature": "", "strict": bool(strict),
    }

    def issue(code, message, path=None, severity="error"):
        entry = {"code": code, "severity": severity, "message": message}
        if path is not None:
            entry["path"] = str(path)
        report["issues"].append(entry)

    try:
        if not config.get("data_dir"):
            raise DatasetError("Set data_dir to the directory containing dung/, duy/ and hung/.")
        owners = _session_splits(config)
        preprocessing = canonical_preprocess_config(config.get("preprocess", {}))
    except (ValueError, TypeError) as exc:
        issue("configuration", str(exc))
        return report
    if not root.is_dir():
        issue("missing_root", "Dataset directory is missing; initialize it and add images to all 18 session folders.", root)
        return report
    unknown_severity = "error" if strict else "warning"
    try:
        for item in sorted(root.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir() and item.name not in CLASSES:
                issue("unexpected_person", "Expected only dung/, duy/ and hung/ directly inside data_dir.", item, unknown_severity)
            elif item.is_file() and item.suffix.lower() in EXTENSIONS:
                issue("misplaced_image", "Move this image into person/phien_XX/ after confirming its label and capture session.", item)
    except OSError as exc:
        issue("unreadable_root", str(exc), root)
        return report

    preprocessor = FacePreprocessor(preprocessing, training=False) if strict and preprocessing["crop_face"] else None
    exact_seen, pixels_seen = {}, {}
    counts = report["counts"]
    for label, person in enumerate(CLASSES):
        person_dir = root / person
        if person_dir.is_symlink():
            issue("symlink_person", "Use a real person directory, not a symbolic link.", person_dir)
            continue
        if person_dir.is_dir():
            for item in sorted(person_dir.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_dir() and item.name not in {f"phien_{n:02d}" for n in SESSIONS}:
                    hint = ("Rename session_XX to phien_XX explicitly after reviewing its contents; no automatic migration is performed."
                            if item.name.startswith("session_") else "Expected session folders phien_01 through phien_06.")
                    issue("unexpected_session", hint, item, unknown_severity)
                elif item.is_file() and item.suffix.lower() in EXTENSIONS:
                    issue("misplaced_image", "Assign this image to its correct phien_XX folder.", item)
        for session in SESSIONS:
            session_name = f"phien_{session:02d}"
            folder = person_dir / session_name
            if folder.is_symlink():
                issue("symlink_session", "Use a real session directory, not a symbolic link.", folder)
                continue
            if not folder.is_dir():
                issue("missing_session", "Required session folder is missing; initialize_dataset creates the 18 folders.", folder)
                continue
            split = owners[session]
            session_count = 0
            for path in sorted(folder.rglob("*")):
                if any(part.startswith(".") for part in path.relative_to(folder).parts):
                    continue
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                if path.is_symlink():
                    issue("symlink_image", "Use a real image file to keep the manifest self-contained.", path)
                    continue
                try:
                    image = load_rgb(path)
                    sha = _sha_file(path)
                    pixel_digest = hashlib.sha256()
                    pixel_digest.update(f"RGB:{image.width}:{image.height}:".encode())
                    pixel_digest.update(image.tobytes())
                    pixel_sha = pixel_digest.hexdigest()
                except (OSError, ValueError) as exc:
                    issue("corrupt_image", str(exc), path)
                    continue
                record = {
                    "path": path.relative_to(root).as_posix(), "sha256": sha,
                    "pixel_sha256": pixel_sha, "label": label, "person": person,
                    "session": session_name, "session_id": session, "split": split,
                    "width": image.width, "height": image.height,
                }
                if preprocessor is not None:
                    try:
                        preprocessor.prepare(image)
                    except (ValueError, RuntimeError, ImportError) as exc:
                        issue("face_validation", str(exc), path)
                # Compare all previous owners: a same-split duplicate must not hide
                # a later copy with a conflicting identity or split.
                priors = exact_seen.get(sha, [])
                duplicate_kind = "exact"
                if not priors:
                    priors = pixels_seen.get(pixel_sha, [])
                    duplicate_kind = "pixels"
                for prior in priors:
                    leakage = prior["split"] != split or prior["person"] != person
                    severity = "error" if leakage else "warning"
                    report["duplicates"].append({"path": record["path"], "matches": prior["path"],
                                                 "kind": duplicate_kind, "severity": severity})
                    issue("duplicate_leakage" if leakage else "duplicate_image",
                          f"{duplicate_kind} duplicate of {prior['path']}; collect independent images for each person/session.",
                          path, severity)
                exact_seen.setdefault(sha, []).append(record)
                pixels_seen.setdefault(pixel_sha, []).append(record)
                report["manifest"].append(record)
                session_count += 1
                counts["total"] += 1
                counts["by_split"][split] += 1
                counts["by_person"][person] += 1
                counts["by_session"][session_name] += 1
                counts["by_person_session"][person][session_name] += 1
                counts["by_split_person"][split][person] += 1
            if not session_count:
                issue("empty_session", "Add at least one valid, independently captured image for this person/session.", folder)
    report["manifest"].sort(key=lambda item: item["path"])
    signature_items = [{"path": item["path"], "sha256": item["sha256"]} for item in report["manifest"]]
    report["signature"] = hashlib.sha256(json.dumps(signature_items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["valid"] = not any(item["severity"] == "error" for item in report["issues"])
    return report


def load_splits(config):
    """Validate the entire dataset, then return fixed session-disjoint samples."""
    report = audit_dataset(config, strict=True)
    if not report["valid"]:
        errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
        details = "; ".join(f"{item['message']} ({item.get('path', '')})" for item in errors[:5])
        raise DatasetError(f"Dataset audit failed ({len(errors)} errors): {details}")
    root = Path(report["root"])
    splits = {name: [] for name in SPLIT_NAMES}
    for item in report["manifest"]:
        splits[item["split"]].append((str(root / item["path"]), item["label"]))
    return splits


class FaceDataset(Dataset):
    def __init__(self, samples, preprocess_config, training=False):
        self.samples = list(samples)
        self.preprocessor = FacePreprocessor(preprocess_config, training=training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.preprocessor(path), int(label)
