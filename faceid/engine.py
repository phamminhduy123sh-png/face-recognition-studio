"""Training and evaluation. Model selection uses validation; test is opt-in."""
import hashlib
import json
import math
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import FaceDataset, load_splits
from .models import ArcFace, build_backbone
from .preprocess import canonical_preprocess_config, preprocess_id

CLASSES = ["dung", "duy", "hung"]
DISPLAY_NAMES = {"dung": "Dũng", "duy": "Duy", "hung": "Hưng"}
FORMAT, VERSION = "face-recognition-studio", 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def resolve_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA không khả dụng; chọn device='cpu'.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Chỉ hỗ trợ cpu, cuda hoặc mps.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS không khả dụng.")
    return device


def checkpoint_path(config, checkpoint=None):
    return Path(checkpoint or config.get("checkpoint") or Path(config["artifacts_dir"]) / "best.pth")


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def sample_signatures(splits, data_dir):
    root = Path(data_dir).resolve()
    records, signatures = {}, {}
    for split, samples in splits.items():
        rows = []
        for path, label in samples:
            path = Path(path)
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                raise ValueError(f"Ảnh nằm ngoài data_dir: {path}")
            rows.append({"path": relative, "label": int(label), "sha256": file_hash(path)})
        rows.sort(key=lambda item: (item["label"], item["path"]))
        records[split] = rows
        signatures[split] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()
    return records, signatures, signature


def _loader(samples, preprocess, settings, training=False):
    return DataLoader(
        FaceDataset(samples, preprocess, training=training),
        batch_size=int(settings.get("batch_size", 16)), shuffle=training,
        num_workers=int(settings.get("num_workers", 0)),
        generator=torch.Generator().manual_seed(int(settings.get("seed", 42))),
        worker_init_fn=seed_worker, pin_memory=False,
    )


@torch.no_grad()
def _prototypes(model, loader, count, embedding_dim, device):
    model.eval()
    sums = torch.zeros(count, embedding_dim, device=device)
    totals = torch.zeros(count, device=device)
    for batch in loader:
        images, labels = batch[:2]
        labels = labels.to(device)
        embeddings = F.normalize(model(images.to(device)), dim=1)
        sums.index_add_(0, labels, embeddings)
        totals.index_add_(0, labels, torch.ones(len(labels), device=device))
    if (totals == 0).any():
        raise ValueError("Train thiếu ảnh của một người; không thể tạo prototype.")
    means = sums / totals[:, None]
    if not torch.isfinite(means).all() or (means.norm(dim=1) < 1e-8).any():
        raise ValueError("Prototype không hợp lệ; kiểm tra dữ liệu và mô hình.")
    return F.normalize(means, dim=1)


@torch.no_grad()
def score_loader(model, centers, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        images, labels = batch[:2]
        embeddings = F.normalize(model(images.to(device)), dim=1)
        scores = (embeddings @ centers.to(device).T).clamp(-1, 1)
        if not torch.isfinite(scores).all():
            raise ValueError("Mô hình sinh điểm không hữu hạn.")
        all_scores.append(scores.cpu())
        all_labels.extend(int(label) for label in labels)
    if not all_scores:
        raise ValueError("Tập đánh giá không có ảnh.")
    return torch.cat(all_scores), torch.tensor(all_labels, dtype=torch.long)


def classification_metrics(scores, labels, classes=CLASSES):
    predictions = scores.argmax(dim=1)
    count = len(classes)
    confusion = torch.zeros(count, count, dtype=torch.int64)
    for true, predicted in zip(labels.tolist(), predictions.tolist()):
        confusion[true, predicted] += 1
    per_class, f1s = {}, []
    for index, name in enumerate(classes):
        true_positive = int(confusion[index, index])
        actual = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return {"count": len(labels), "accuracy_known_only": float((predictions == labels).float().mean()),
            "macro_f1": sum(f1s) / count, "confusion_matrix": confusion.tolist(),
            "confusion_convention": "rows=true, columns=predicted", "classes": list(classes),
            "per_class": per_class}


def _cpu_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _save_bundle(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, path)


def _archive_previous_run(output):
    filenames = ("best.pth", "history.json", "training_summary.json", "train_metrics.json",
                 "val_metrics.json", "test_metrics.json", "threshold.json")
    previous = [output / name for name in filenames if (output / name).is_file()]
    if not previous:
        return None
    destination = output / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    for source in previous:
        shutil.move(str(source), str(destination / source.name))
    return str(destination)


def train(config, progress=None):
    settings = dict(config.get("training", {}))
    epochs, patience = int(settings.get("epochs", 40)), int(settings.get("patience", 8))
    if epochs < 1 or patience < 1 or int(settings.get("batch_size", 16)) < 1:
        raise ValueError("epochs, patience, batch_size phải >= 1.")
    if int(settings.get("num_workers", 0)) < 0:
        raise ValueError("num_workers phải >= 0.")
    lr, weight_decay = float(settings.get("lr", 0.0003)), float(settings.get("weight_decay", 0.0001))
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("lr phải > 0 và weight_decay phải >= 0.")
    seed = int(settings.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(settings.get("device", "auto"))
    splits = load_splits(config)
    if not splits.get("train") or not splits.get("val"):
        raise ValueError("Cần train và validation riêng theo phiên trước khi huấn luyện.")
    records, signatures, signature = sample_signatures(splits, config["data_dir"])
    preprocess = canonical_preprocess_config(config.get("preprocess", {}))
    input_mode = config.get("input_mode", "detect_faces")
    if input_mode not in {"raw_image", "detect_faces"}:
        raise ValueError("input_mode phải là raw_image hoặc detect_faces.")
    if input_mode == "raw_image" and preprocess["crop_face"]:
        raise ValueError("input_mode raw_image yêu cầu preprocess.crop_face=False.")
    architecture, dimension = settings.get("backbone", "resnet18"), int(settings.get("embedding_dim", 128))
    warm_start = settings.get("warm_start")
    if warm_start and architecture not in {"facecnn", "legacy"}:
        raise ValueError("warm_start backbone cũ chỉ dùng với backbone='facecnn'.")
    model = build_backbone(architecture, dimension, bool(settings.get("pretrained", True))).to(device)
    warm_start_sha256 = None
    if warm_start:
        warm_start_path = Path(warm_start).expanduser()
        if not warm_start_path.is_absolute() and config.get("config_path"):
            warm_start_path = Path(config["config_path"]).resolve().parent / warm_start_path
        warm_start_path = warm_start_path.resolve()
        settings["warm_start"] = str(warm_start_path)
        state = torch.load(warm_start_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "backbone" in state:
            state = state["backbone"]
        model.load_state_dict(state, strict=True)
        warm_start_sha256 = file_hash(warm_start_path)
    margin, scale = float(settings.get("margin", 0.3)), float(settings.get("scale", 30.0))
    head = ArcFace(dimension, len(CLASSES), margin, scale).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    train_loader = _loader(splits["train"], preprocess, settings, training=True)
    deterministic_train = _loader(splits["train"], preprocess, settings, training=False)
    val_loader = _loader(splits["val"], preprocess, settings, training=False)
    output = Path(config["artifacts_dir"])
    output.mkdir(parents=True, exist_ok=True)
    archived_previous_run = _archive_previous_run(output)
    destination = output / "best.pth"
    history, best_key, stale = [], (-1.0, -1.0), 0
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        loss_sum, examples = 0.0, 0
        for batch in train_loader:
            images, labels = batch[:2]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(model(images), labels), labels)
            if not torch.isfinite(loss):
                raise ValueError("Loss không hữu hạn; huấn luyện đã dừng.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            examples += len(labels)
        centers = _prototypes(model, deterministic_train, len(CLASSES), dimension, device)
        scores, labels = score_loader(model, centers, val_loader, device)
        val_metrics = classification_metrics(scores, labels)
        key = (val_metrics["accuracy_known_only"], val_metrics["macro_f1"])
        scheduler.step(val_metrics["macro_f1"])
        record = {"epoch": epoch, "train_loss": loss_sum / examples, "val": val_metrics,
                  "lr": optimizer.param_groups[0]["lr"], "epochs": epochs,
                  "progress": 100 * epoch / epochs, "loss": loss_sum / examples,
                  "val_accuracy": val_metrics["accuracy_known_only"],
                  "message": f"Epoch {epoch}/{epochs} | loss {loss_sum / examples:.4f} | val {val_metrics['accuracy_known_only']:.1%}"}
        history.append(record)
        if key > best_key:
            best_key, stale = key, 0
            bundle = {"format": FORMAT, "version": VERSION, "architecture": architecture,
                      "embedding_dim": dimension, "classes": CLASSES.copy(), "preprocess": preprocess,
                      "preprocess_id": preprocess_id(preprocess), "backbone": _cpu_state(model),
                      "head": _cpu_state(head), "arcface": {"margin": margin, "scale": scale},
                      "prototypes": centers.detach().cpu(), "prototype_source": "train_deterministic_only",
                      "splits": config.get("splits", {}), "data_signature": signature,
                      "split_signatures": signatures, "train_hashes": [row["sha256"] for row in records["train"]],
                      "data_manifest": records, "metrics": {"val": val_metrics}, "epoch": epoch,
                      "training": settings, "validation_status": "known_validation_only",
                      "warm_start_sha256": warm_start_sha256,
                      "source": "trained", "test_used_for_selection": False,
                      "input_mode": input_mode}
            _save_bundle(destination, bundle)
        else:
            stale += 1
        write_json(output / "history.json", history)
        if progress:
            progress(record)
        if stale >= patience:
            break
    best = torch.load(destination, map_location="cpu", weights_only=True)
    summary = {"checkpoint": str(destination), "epochs_completed": len(history), "best_epoch": best["epoch"],
               "architecture": architecture, "classes": CLASSES.copy(), "metrics": best["metrics"],
               "preprocess_id": best["preprocess_id"], "split_counts": {name: len(rows) for name, rows in splits.items()},
               "test_used_for_selection": False, "threshold_calibrated": False,
               "archived_previous_run": archived_previous_run}
    write_json(output / "training_summary.json", summary)
    return summary


def validate_dataset_signature(bundle, splits, config, split=None):
    _, signatures, _ = sample_signatures(splits, config["data_dir"])
    previous = bundle.get("split_signatures", {})
    if previous:
        names = [split] if split else list(previous)
        for name in names:
            if previous.get(name) != signatures.get(name):
                raise ValueError(f"Dữ liệu {name} đã thay đổi so với checkpoint; cần huấn luyện/hiệu chỉnh lại.")
    return signatures


def evaluate(config, split="test", checkpoint=None):
    if split not in {"train", "val", "test"}:
        raise ValueError("split phải là train, val hoặc test.")
    from .inference import load_bundle, load_threshold
    path = checkpoint_path(config, checkpoint)
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    model, bundle = load_bundle(path, device)
    audit_config = {**config, "preprocess": bundle["preprocess"]}
    splits = load_splits(audit_config)
    validate_dataset_signature(bundle, splits, config)
    if not splits.get(split):
        raise ValueError(f"Tập {split} không có ảnh.")
    scores, labels = score_loader(model, bundle["prototypes"],
                                  _loader(splits[split], bundle["preprocess"], config.get("training", {})), device)
    report = classification_metrics(scores, labels, bundle["classes"])
    report.update({"split": split, "checkpoint": str(path), "checkpoint_sha256": bundle["_checkpoint_sha256"],
                   "preprocess_id": bundle["preprocess_id"], "open_set": None,
                   "validation_status": bundle.get("validation_status", "unvalidated")})
    threshold_path = Path(config.get("threshold_file") or Path(config["artifacts_dir"]) / "threshold.json")
    if threshold_path.is_file():
        threshold, _ = load_threshold(threshold_path, path, bundle)
        best_scores, predictions = scores.max(dim=1)
        # Keep the float64 nextafter boundary used during calibration.
        accepted = best_scores.double() >= threshold
        report["open_set"] = {"threshold": threshold, "accepted": int(accepted.sum()),
                              "rejected": int((~accepted).sum()), "coverage": float(accepted.float().mean()),
                              "correct_and_accepted_rate": float(((predictions == labels) & accepted).float().mean()),
                              "known_false_reject_rate": float((~accepted).float().mean()),
                              "note": "Tập này chỉ có người đã đăng ký; không đo FAR người lạ ở đây."}
    write_json(Path(config["artifacts_dir"]) / f"{split}_metrics.json", report)
    return report


def import_legacy(config):
    legacy = Path(config["legacy_dir"])
    names = [line.strip() for line in (legacy / "class_names.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if names != CLASSES:
        raise ValueError(f"Thứ tự nhãn weights cũ phải là {CLASSES}; hiện là {names}.")
    backbone_state = torch.load(legacy / "arcface_backbone.pth", map_location="cpu", weights_only=True)
    head_state = torch.load(legacy / "arcface_head.pth", map_location="cpu", weights_only=True)
    dimension = int(backbone_state["embedding_layer.weight"].shape[0])
    model = build_backbone("facecnn", dimension, pretrained=False)
    model.load_state_dict(backbone_state, strict=True)
    head = ArcFace(dimension, len(CLASSES))
    head.load_state_dict(head_state, strict=True)
    if not all(torch.isfinite(t).all() for t in list(backbone_state.values()) + list(head_state.values())):
        raise ValueError("Weights cũ chứa giá trị không hữu hạn.")
    if (head_state["weight"].norm(dim=1) < 1e-8).any():
        raise ValueError("ArcFace head có vector lớp bằng zero.")
    preprocess = {"size": [112, 112], "resize_mode": "resize", "normalization": "none",
                  "crop_face": False, "face_margin": 0.0, "no_face": "error"}
    warning = "Weights cũ chưa được kiểm chứng trên validation/test theo phiên mới; điểm cosine không phải xác suất."
    bundle = {"format": FORMAT, "version": VERSION, "architecture": "facecnn", "embedding_dim": dimension,
              "classes": names, "preprocess": preprocess, "preprocess_id": preprocess_id(preprocess),
              "backbone": backbone_state, "head": head_state, "prototypes": F.normalize(head_state["weight"], dim=1),
              "prototype_source": "legacy_arcface_head_weights", "source": "legacy_import",
              "legacy_sha256": {name: file_hash(legacy / name) for name in
                                ("arcface_backbone.pth", "arcface_head.pth", "class_names.txt")},
              "validation_status": "unvalidated_legacy", "warnings": [warning], "splits": {},
              "input_mode": "detect_faces",
              "data_signature": None, "split_signatures": {}, "train_hashes": [], "metrics": {}, "epoch": None}
    destination = Path(config["artifacts_dir"]) / "legacy.pth"
    _save_bundle(destination, bundle)
    summary = {"checkpoint": str(destination), "architecture": "facecnn", "embedding_dim": dimension,
               "classes": names, "preprocess": preprocess, "validation_status": "unvalidated_legacy", "warnings": [warning]}
    write_json(Path(config["artifacts_dir"]) / "legacy_import.json", summary)
    return summary
