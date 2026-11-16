"""Cấu hình duy nhất, đường dẫn tương đối tính từ file JSON."""
from copy import deepcopy
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["dung", "duy", "hung"]
DISPLAY_NAMES = {"dung": "Dũng", "duy": "Duy", "hung": "Hưng"}
DEFAULTS = {
    "data_dir": "data/faces", "artifacts_dir": "artifacts",
    "legacy_dir": "weights/legacy",
    "input_mode": "detect_faces",
    "splits": {"train": [1, 2, 3, 4], "val": [5], "test": [6]},
    "preprocess": {"size": [224, 224], "resize_mode": "pad",
                   "normalization": "imagenet", "crop_face": True,
                   "face_margin": 0.2, "no_face": "error"},
    "training": {"backbone": "resnet18", "pretrained": True,
                 "embedding_dim": 128, "epochs": 40, "batch_size": 16,
                 "lr": 0.0003, "weight_decay": 0.0001,
                 "patience": 8, "seed": 42, "num_workers": 0,
                 "device": "auto"},
}


def _merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    default_path = PROJECT_ROOT / "config.json"
    if not default_path.is_file():
        default_path = Path.cwd() / "config.json"
    config_path = Path(path or default_path).resolve()
    if path is not None and not config_path.is_file():
        raise ValueError(f"Không tìm thấy cấu hình: {config_path}")
    values = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if not isinstance(values, dict):
        raise ValueError("File cấu hình phải là JSON object.")
    config = _merge(deepcopy(DEFAULTS), values)
    config["config_path"] = str(config_path)
    for key in ("data_dir", "artifacts_dir", "legacy_dir", "checkpoint"):
        if config.get(key):
            p = Path(config[key]).expanduser()
            config[key] = str((p if p.is_absolute() else config_path.parent / p).resolve())
    training = config["training"]
    for key in ("epochs", "batch_size", "patience", "embedding_dim"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} phải là số nguyên > 0.")
    if type(training["num_workers"]) is not int or training["num_workers"] < 0:
        raise ValueError("training.num_workers phải là số nguyên >= 0.")
    for key in ("lr", "weight_decay"):
        if not math.isfinite(training[key]) or training[key] < 0 or (key == "lr" and training[key] == 0):
            raise ValueError(f"training.{key} không hợp lệ.")
    return config


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
