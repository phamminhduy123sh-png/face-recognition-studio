"""Calibrate a cosine rejection threshold on known/unknown VALIDATION only."""
import hashlib
import json
import math
from pathlib import Path

import torch

from .data import EXTENSIONS, load_splits
from .engine import (_loader, checkpoint_path, file_hash, resolve_device, score_loader,
                     validate_dataset_signature, write_json)
from .inference import load_bundle
from .preprocess import load_rgb


def _pixel_hash(image):
    digest = hashlib.sha256()
    digest.update(f"RGB:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _unknown_validation(unknown_dir, splits, config):
    directory = Path(unknown_dir).resolve()
    data_root = Path(config["data_dir"]).resolve()
    if not directory.is_dir():
        raise ValueError(f"Không có thư mục validation người lạ: {directory}")
    if directory == data_root or directory.is_relative_to(data_root):
        raise ValueError("Validation người lạ phải nằm riêng ngoài data_dir của ba người đã đăng ký.")
    known_paths, known_hashes, known_pixels = set(), set(), set()
    # Test data are hashed only to prevent leakage; never scored or used to choose threshold.
    for samples in splits.values():
        for path, _ in samples:
            path = Path(path)
            known_paths.add(path.resolve())
            known_hashes.add(file_hash(path))
            known_pixels.add(_pixel_hash(load_rgb(path)))
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in EXTENSIONS
                   and not any(part.startswith(".") for part in path.relative_to(directory).parts))
    if not paths:
        raise ValueError("Validation người lạ chưa có ảnh.")
    unknown_hashes, unknown_pixels, records = set(), set(), []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError(f"Ảnh người lạ phải là file thật trong unknown_dir: {path}")
        sha, pixels = file_hash(path), _pixel_hash(load_rgb(path))
        if path.resolve() in known_paths or sha in known_hashes or pixels in known_pixels:
            raise ValueError(f"Ảnh người lạ trùng ảnh trong train/val/test: {path}")
        if sha in unknown_hashes or pixels in unknown_pixels:
            raise ValueError(f"Ảnh người lạ bị lặp; loại bản sao trước khi hiệu chỉnh: {path}")
        unknown_hashes.add(sha)
        unknown_pixels.add(pixels)
        records.append({"path": path.relative_to(directory).as_posix(), "sha256": sha})
    signature = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return [(str(path), 0) for path in paths], records, signature


def choose_threshold(known_scores, known_correct, unknown_scores, max_far=0.05):
    """Maximize known correct accepts under an empirical unknown FAR bound."""
    max_far = float(max_far)
    if not math.isfinite(max_far) or not 0 <= max_far <= 1:
        raise ValueError("max_far phải nằm trong [0,1].")
    known = [float(value) for value in known_scores]
    unknown = [float(value) for value in unknown_scores]
    correct = [bool(value) for value in known_correct]
    if not known or not unknown or len(correct) != len(known):
        raise ValueError("Cần điểm validation của cả người quen và người lạ.")
    if not all(math.isfinite(value) and -1 <= value <= 1 for value in known + unknown):
        raise ValueError("Điểm cosine không hợp lệ.")
    candidates = sorted({-1.0, math.nextafter(1.0, math.inf), *known, *unknown,
                         *(math.nextafter(value, math.inf) for value in known + unknown)})
    best = None
    for threshold in candidates:
        unknown_accepts = sum(score >= threshold for score in unknown)
        far = unknown_accepts / len(unknown)
        if far > max_far + 1e-12:
            continue
        known_accepts = sum(score >= threshold for score in known)
        correct_accepts = sum(ok and score >= threshold for ok, score in zip(correct, known))
        wrong_accepts = known_accepts - correct_accepts
        key = (correct_accepts, -far, -wrong_accepts, threshold)
        if best is None or key > best[0]:
            best = (key, {"threshold": threshold, "empirical_unknown_far": far,
                          "unknown_accepted": unknown_accepts, "unknown_count": len(unknown),
                          "known_count": len(known), "known_accepted": known_accepts,
                          "known_wrong_accepted": wrong_accepts,
                          "known_coverage": known_accepts / len(known),
                          "known_false_reject_rate": 1 - known_accepts / len(known),
                          "known_correct_and_accepted_rate": correct_accepts / len(known),
                          "known_closed_set_accuracy": sum(correct) / len(correct)})
    return best[1]


def calibrate(config, unknown_dir, max_far=0.05, checkpoint=None):
    max_far = float(max_far)
    if not math.isfinite(max_far) or not 0 <= max_far <= 1:
        raise ValueError("max_far phải nằm trong [0,1].")
    path = checkpoint_path(config, checkpoint)
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    model, bundle = load_bundle(path, device)
    # Use the checkpoint's pipeline, even when the main config has since changed.
    audit_config = {**config, "preprocess": bundle["preprocess"]}
    splits = load_splits(audit_config)
    signatures = validate_dataset_signature(bundle, splits, config)
    if not splits.get("val"):
        raise ValueError("Không có validation người đã đăng ký.")
    unknown, unknown_manifest, unknown_signature = _unknown_validation(unknown_dir, splits, config)
    settings = config.get("training", {})
    known_scores, labels = score_loader(model, bundle["prototypes"],
                                       _loader(splits["val"], bundle["preprocess"], settings), device)
    unknown_scores, _ = score_loader(model, bundle["prototypes"],
                                    _loader(unknown, bundle["preprocess"], settings), device)
    best_known, predictions = known_scores.max(dim=1)
    report = choose_threshold(best_known.tolist(), (predictions == labels).tolist(),
                              unknown_scores.max(dim=1).values.tolist(), max_far)
    warnings = ["FAR là tỷ lệ thực nghiệm trên validation người lạ, không bảo đảm trên người chưa gặp.",
                "Ảnh người lạ phải được chụp độc lập và có định dạng đầu vào giống validation người quen."]
    if report["unknown_count"] < 20:
        warnings.append("Có ít hơn 20 ảnh người lạ; ước lượng FAR còn rất hạn chế.")
    if report["known_coverage"] == 0:
        warnings.append("Ngưỡng hiện từ chối toàn bộ người quen; cần thêm dữ liệu hoặc huấn luyện lại mô hình.")
    if bundle.get("source") == "legacy_import":
        warnings.append("Weights cũ chưa có lịch sử train để kiểm chứng độc lập với validation mới.")
    report.update({"format": "face-recognition-threshold", "version": 1,
                   "checkpoint": str(path), "checkpoint_sha256": bundle["_checkpoint_sha256"],
                   "preprocess_id": bundle["preprocess_id"], "classes": bundle["classes"],
                   "calibrated_on": "validation_only", "max_far_requested": max_far,
                   "known_validation_signature": signatures["val"],
                   "unknown_validation_dir": str(Path(unknown_dir).resolve()),
                   "unknown_validation_signature": unknown_signature,
                   "unknown_validation_manifest": unknown_manifest,
                   "test_used_for_threshold": False, "score_type": "cosine_similarity_not_probability",
                   "warnings": warnings})
    destination = Path(config["artifacts_dir"]) / "threshold.json"
    write_json(destination, report)
    return {**report, "threshold_file": str(destination)}
