"""One shared, versioned preprocessing pipeline for training and inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

_DEFAULTS = {
    "size": [112, 112], "resize_mode": "pad", "normalization": "none",
    "crop_face": True, "face_margin": 0.2, "no_face": "error",
    "augment": True, "horizontal_flip": True,
}


def canonical_preprocess_config(config=None):
    """Validate options and supply stable defaults; size is [height, width]."""
    config = {} if config is None else dict(config)
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown preprocessing options: {sorted(unknown)}")
    options = {**_DEFAULTS, **config}
    size = options["size"]
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in size)):
        raise ValueError("preprocess.size must contain two positive integers [height, width].")
    options["size"] = list(size)
    if options["resize_mode"] not in {"pad", "resize"}:
        raise ValueError("preprocess.resize_mode must be 'pad' or 'resize'.")
    if options["normalization"] not in {"none", "imagenet", "symmetric"}:
        raise ValueError("preprocess.normalization must be none, imagenet or symmetric.")
    if options["no_face"] not in {"error", "keep"}:
        raise ValueError("preprocess.no_face must be 'error' or 'keep'.")
    margin = options["face_margin"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("preprocess.face_margin must be finite and >= 0.")
    options["face_margin"] = float(margin)
    for name in ("crop_face", "augment", "horizontal_flip"):
        if not isinstance(options[name], bool):
            raise ValueError(f"preprocess.{name} must be true or false.")
    return options


def preprocess_id(config=None):
    """Portable fingerprint includes defaults and a pipeline implementation version."""
    payload = {"pipeline_version": 1, **canonical_preprocess_config(config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_rgb(image):
    """Decode, honor EXIF and composite transparent pixels against white."""
    try:
        if isinstance(image, Image.Image):
            oriented = ImageOps.exif_transpose(image).convert("RGBA")
        else:
            with Image.open(Path(image)) as source:
                oriented = ImageOps.exif_transpose(source).convert("RGBA")
        white = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, oriented).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot decode image: {image}") from exc


class FaceDetector:
    """Offline Haar detector; OpenCV/cascade initialized only when first needed."""
    def __init__(self):
        self._cascade = None

    def __getstate__(self):
        return {"_cascade": None}

    def detect(self, image):
        if self._cascade is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Face detection requires: pip install opencv-python-headless") from exc
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self._cascade.empty():
                raise RuntimeError("Could not load OpenCV Haar face cascade.")
        boxes = self._cascade.detectMultiScale(
            np.asarray(image.convert("L")),
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )

        # Chuyển bounding box thành tuple số nguyên
        boxes = sorted(
            [tuple(map(int, box)) for box in boxes],
            key=lambda box: box[2] * box[3],
            reverse=True
        )

        if not boxes:
            return []

        # Lấy diện tích bounding box lớn nhất
        largest_area = boxes[0][2] * boxes[0][3]

        # Loại bỏ những vùng quá nhỏ so với khuôn mặt lớn nhất
        filtered_boxes = [
            box for box in boxes
            if box[2] * box[3] >= largest_area * 0.15
        ]

        return filtered_boxes

    @staticmethod
    def crop(image, bbox, margin=0.2):
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("Face margin must be finite and >= 0.")
        if len(bbox) != 4:
            raise ValueError("Face bounding box must contain x, y, width, height.")
        x, y, width, height = map(int, bbox)
        if width <= 0 or height <= 0:
            raise ValueError("Face bounding box dimensions must be positive.")
        dx, dy = round(width * margin), round(height * margin)
        left, top = max(0, x - dx), max(0, y - dy)
        right, bottom = min(image.width, x + width + dx), min(image.height, y + height + dy)
        if right <= left or bottom <= top:
            raise ValueError("Face bounding box lies outside the image.")
        return image.crop((left, top, right, bottom))


class FacePreprocessor:
    def __init__(self, config=None, training=False):
        self.config = canonical_preprocess_config(config)
        self.training = bool(training)
        self._detector = None
        steps = []
        if self.training and self.config["augment"]:
            if self.config["horizontal_flip"]:
                steps.append(transforms.RandomHorizontalFlip(p=0.5))
            steps.extend([
                transforms.RandomRotation(10, fill=(255, 255, 255)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ])
        steps.append(transforms.ToTensor())
        if self.config["normalization"] == "imagenet":
            steps.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif self.config["normalization"] == "symmetric":
            steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(steps)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_detector"] = None
        return state

    @property
    def detector(self):
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    def prepare(self, image, already_cropped=False):
        image = load_rgb(image)
        if self.config["crop_face"] and not already_cropped:
            boxes = self.detector.detect(image)
            if not boxes:
                if self.config["no_face"] == "error":
                    raise ValueError(
                        "No face detected. Retake the image or explicitly set preprocess.no_face='keep'."
                    )
            else:
                # detect() đã sắp xếp bounding box theo diện tích giảm dần
                main_face = boxes[0]

                image = self.detector.crop(
                    image,
                    main_face,
                    margin=self.config["face_margin"]
                )
        height, width = self.config["size"]
        if self.config["resize_mode"] == "pad":
            return ImageOps.pad(image, (width, height), method=Image.Resampling.BILINEAR, color="white")
        return image.resize((width, height), Image.Resampling.BILINEAR)

    def tensor(self, image, already_cropped=False):
        return self.transform(self.prepare(image, already_cropped=already_cropped))

    def __call__(self, image, already_cropped=False):
        return self.tensor(image, already_cropped=already_cropped)
