"""Checkpoint-bound cosine matching; no threshold implies no accepted identity."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .engine import CLASSES, DISPLAY_NAMES, FORMAT, VERSION, file_hash, resolve_device
from .models import build_backbone
from .preprocess import FaceDetector, FacePreprocessor, canonical_preprocess_config, load_rgb, preprocess_id


def load_bundle(checkpoint, device="cpu"):
    device = resolve_device(str(device))
    # Hash and deserialize the same opened inode: retraining atomically replaces
    # best.pth while a browser request may still be loading the previous model.
    with Path(checkpoint).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        checkpoint_sha256 = digest.hexdigest()
        stream.seek(0)
        bundle = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT or bundle.get("version") != VERSION:
        raise ValueError("Checkpoint không đúng định dạng; dùng import-legacy cho weights rời.")
    if bundle.get("classes") != CLASSES:
        raise ValueError(f"Checkpoint phải có thứ tự lớp {CLASSES}.")
    dimension = bundle.get("embedding_dim")
    if not isinstance(dimension, int) or dimension < 2:
        raise ValueError("embedding_dim trong checkpoint không hợp lệ.")
    preprocess = canonical_preprocess_config(bundle.get("preprocess", {}))
    if bundle.get("preprocess_id") != preprocess_id(preprocess):
        raise ValueError("preprocess_id không khớp metadata checkpoint.")
    # Old bundles without this field assume face crops, matching their recognizer.
    input_mode = bundle.get("input_mode", "detect_faces")
    if input_mode not in {"raw_image", "detect_faces"}:
        raise ValueError("input_mode checkpoint không hợp lệ.")
    if input_mode == "raw_image" and preprocess["crop_face"]:
        raise ValueError("raw_image không được bật crop_face; metadata mâu thuẫn.")
    centers = bundle.get("prototypes")
    if not isinstance(centers, torch.Tensor) or centers.shape != (len(CLASSES), dimension):
        raise ValueError("Số lượng hoặc kích thước prototype không khớp lớp/embedding.")
    if not torch.isfinite(centers).all() or (centers.norm(dim=1) < 1e-8).any():
        raise ValueError("Prototype chứa giá trị không hợp lệ.")
    model = build_backbone(bundle.get("architecture"), dimension, pretrained=False)
    try:
        model.load_state_dict(bundle["backbone"], strict=True)
    except (RuntimeError, KeyError, TypeError) as exc:
        raise ValueError("Weights backbone không khớp kiến trúc checkpoint.") from exc
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Weights chứa giá trị không hữu hạn.")
    bundle["preprocess"] = preprocess
    bundle["_checkpoint_sha256"] = checkpoint_sha256
    bundle["input_mode"] = input_mode
    bundle["prototypes"] = F.normalize(centers.float(), dim=1)
    return model.to(device).eval(), bundle


def load_threshold(path, checkpoint, bundle):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("File threshold phải là JSON object.")
    expected_sha256 = bundle.get("_checkpoint_sha256") or file_hash(checkpoint)
    if record.get("checkpoint_sha256") != expected_sha256:
        raise ValueError("Ngưỡng thuộc checkpoint khác; cần hiệu chỉnh lại.")
    if record.get("preprocess_id") != bundle["preprocess_id"] or record.get("classes") != bundle["classes"]:
        raise ValueError("Ngưỡng không khớp tiền xử lý hoặc thứ tự người.")
    try:
        value = float(record["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("File threshold thiếu giá trị số hợp lệ.") from exc
    if not math.isfinite(value) or not -1 <= value <= math.nextafter(1.0, math.inf):
        raise ValueError("Ngưỡng cosine không hợp lệ.")
    return value, record


class Predictor:
    def __init__(self, checkpoint, threshold_file=None, device="auto", manual_threshold=None):
        if threshold_file is not None and manual_threshold is not None:
            raise ValueError("Chọn threshold_file hoặc manual_threshold, không dùng đồng thời.")
        self.checkpoint = Path(checkpoint)
        self.device = resolve_device(device)
        self.model, self.bundle = load_bundle(self.checkpoint, self.device)
        self.classes = self.bundle["classes"]
        self.centers = self.bundle["prototypes"].to(self.device)
        self.preprocessor = FacePreprocessor(self.bundle["preprocess"], training=False)
        self.detector = FaceDetector()
        self.threshold, self.threshold_record = None, None
        self.threshold_source = "none"
        if threshold_file is not None:
            self.threshold, self.threshold_record = load_threshold(threshold_file, self.checkpoint, self.bundle)
            self.threshold_source = "calibrated"
        elif manual_threshold is not None:
            value = float(manual_threshold)
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("Ngưỡng thử thủ công phải nằm trong [-1,1].")
            self.threshold = value
            self.threshold_source = "manual"
        self.metadata = {
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.bundle["_checkpoint_sha256"],
            "architecture": self.bundle["architecture"], "embedding_dim": self.bundle["embedding_dim"],
            "classes": self.classes, "preprocess": self.bundle["preprocess"],
            "preprocess_id": self.bundle["preprocess_id"], "input_mode": self.bundle["input_mode"],
            "source": self.bundle.get("source", "unknown"),
            "validation_status": self.bundle.get("validation_status", "unvalidated"),
            "metrics": self.bundle.get("metrics", {}), "warnings": self.bundle.get("warnings", []),
            "threshold": self.threshold, "threshold_source": self.threshold_source,
            "calibrated": self.threshold_source == "calibrated", "score_type": "cosine_similarity_not_probability",
        }

    @torch.inference_mode()
    def _predict(self, image, already_cropped):
        tensor = self.preprocessor.tensor(image, already_cropped=already_cropped).unsqueeze(0).to(self.device)
        embedding = F.normalize(self.model(tensor), dim=1)
        scores = (embedding @ self.centers.T).clamp(-1, 1)[0].cpu()
        if not torch.isfinite(scores).all():
            raise ValueError("Mô hình sinh điểm không hữu hạn.")
        index, score = int(scores.argmax()), float(scores.max())
        candidate = self.classes[index]
        accepted = self.threshold is not None and score >= self.threshold
        calibrated = self.threshold_source == "calibrated"
        if self.threshold is None:
            person, display_name, status = None, "Chưa hiệu chỉnh", "uncalibrated"
        elif not accepted:
            person, display_name, status = "unknown", "Không xác định", "unknown"
        else:
            person, display_name = candidate, DISPLAY_NAMES[candidate]
            status = "recognized" if calibrated else "manual_match"
        return {"candidate": candidate, "candidate_display_name": DISPLAY_NAMES[candidate],
                "person": person, "display_name": display_name, "score": score,
                "scores": {name: float(scores[i]) for i, name in enumerate(self.classes)},
                "accepted": bool(accepted), "status": status, "calibrated": calibrated,
                "threshold": self.threshold, "threshold_source": self.threshold_source,
                "score_type": "cosine_similarity_not_probability"}

    def predict_face(self, image):
        """Input is explicitly already cropped; never run face detection again."""
        return self._predict(image, already_cropped=True)

    def predict(self, image):
        image = load_rgb(image)
        if self.bundle["input_mode"] == "raw_image":
            # Full-image training must use full-image inference, including geometry.
            faces = [{"bbox": [0, 0, image.width, image.height],
                      **self._predict(image, already_cropped=False)}]
        else:
            faces = []
            for box in self.detector.detect(image):
                cropped = self.detector.crop(image, box, self.bundle["preprocess"]["face_margin"])
                faces.append({"bbox": list(box), **self.predict_face(cropped)})
        return {"faces": faces, "face_count": len(faces), "image_size": [image.width, image.height],
                "input_mode": self.bundle["input_mode"], "calibrated": self.threshold_source == "calibrated",
                "threshold_source": self.threshold_source, "threshold": self.threshold,
                "score_type": "cosine_similarity_not_probability"}
