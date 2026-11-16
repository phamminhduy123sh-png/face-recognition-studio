"""Local FastAPI interface. Run through ``python -m faceid serve``."""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import os
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
PEOPLE = {"dung": "Dũng", "duy": "Duy", "hung": "Hưng"}
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}


class TrainRequest(BaseModel):
    epochs: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=256)
    backbone: str = "resnet18"


class CalibrationRequest(BaseModel):
    max_far: float = Field(default=0.05, ge=0, le=1)


def create_app(config_path=None, services=None):
    """Create an app; optional service overrides support camera-free tests."""
    overrides = dict(services or {})

    def service(key, module, name):
        if key in overrides:
            return overrides[key]
        return getattr(importlib.import_module(module), name)

    config = overrides.get("config")
    if config is None:
        config = service("load_config", "faceid.config", "load_config")(
            config_path
        )
    config = copy.deepcopy(config)
    data_dir = Path(config["data_dir"])
    artifacts_dir = Path(config["artifacts_dir"])
    app = FastAPI(title="Face Recognition Studio", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.services = overrides
    app.state.job = None
    app.state.lock = threading.Lock()
    app.state.predictor = None
    app.state.predictor_key = None
    app.state.inference_lock = threading.Lock()
    app.state.hash_cache = {}

    @app.exception_handler(Exception)
    async def friendly_error(request, exc):
        return JSONResponse(status_code=500, content={
            "detail": "Không thể hoàn tất thao tác. " + str(exc)
        })

    def checkpoint():
        for name in ("best.pth", "legacy.pth"):
            candidate = artifacts_dir / name
            if candidate.is_file():
                return candidate
        return None

    def threshold_file():
        for name in ("threshold.json", "thresholds.json", "calibration.json"):
            candidate = artifacts_dir / name
            if candidate.is_file():
                return candidate
        return None

    async def read_image(upload):
        raw = await upload.read(MAX_UPLOAD + 1)
        await upload.close()
        if not raw:
            raise HTTPException(400, "Ảnh tải lên đang trống.")
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, "Ảnh vượt quá 10 MB. Hãy chọn ảnh nhỏ hơn.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as original:
                    fmt = original.format
                    if fmt not in FORMATS:
                        raise HTTPException(400, "Chỉ nhận ảnh JPG, PNG, WEBP hoặc BMP.")
                    if original.width * original.height > MAX_PIXELS:
                        raise HTTPException(413, "Ảnh quá lớn. Giới hạn là 24 triệu pixel.")
                    original.load()
                    rgba = ImageOps.exif_transpose(original).convert("RGBA")
                    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(white, rgba).convert("RGB")
        except HTTPException:
            raise
        except (OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise HTTPException(400, "Không đọc được ảnh. Hãy dùng một ảnh hợp lệ.") from exc
        return raw, image, FORMATS[fmt]

    def get_predictor():
        model = checkpoint()
        if model is None:
            raise HTTPException(409, "Chưa có mô hình. Hãy huấn luyện hoặc nhập trọng số trước.")
        threshold = threshold_file()
        infer = config.get("inference", {})
        key = (str(model), model.stat().st_mtime_ns,
               str(threshold), threshold.stat().st_mtime_ns if threshold else None,
               infer.get("manual_threshold"))
        with app.state.lock:
            if app.state.predictor_key != key:
                predictor_cls = service("Predictor", "faceid.inference", "Predictor")
                app.state.predictor = predictor_cls(
                    model, threshold_file=threshold,
                    device=infer.get("device", "auto"),
                    manual_threshold=infer.get("manual_threshold"),
                )
                app.state.predictor_key = key
            return app.state.predictor

    def current_job():
        with app.state.lock:
            return copy.deepcopy(app.state.job)

    def threshold_ready(model):
        threshold = threshold_file()
        if not model or not threshold:
            return False
        try:
            info = json.loads(threshold.read_text(encoding="utf-8"))
            expected = info.get("checkpoint_sha256")
            if not expected or not isinstance(info.get("threshold"), (int, float)):
                return False
            key = (str(model), model.stat().st_mtime_ns, model.stat().st_size)
            if key not in app.state.hash_cache:
                with model.open("rb") as stream:
                    app.state.hash_cache = {key: hashlib.file_digest(stream, "sha256").hexdigest()}
            return expected == app.state.hash_cache[key]
        except (OSError, ValueError):
            return False

    def start_job(kind, operation):
        with app.state.lock:
            if app.state.job and app.state.job["status"] in {"queued", "running"}:
                raise HTTPException(409, "Một tác vụ đang chạy. Vui lòng chờ hoàn tất.")
            job = dict(id=uuid.uuid4().hex, kind=kind, status="queued", progress=0,
                       logs=[], started_at=datetime.now(timezone.utc).isoformat(),
                       finished_at=None, result=None, error=None)
            app.state.job = job

        def progress(event=None, **kwargs):
            payload = dict(event) if isinstance(event, dict) else {"message": str(event or "")}
            payload.update(kwargs)
            message = payload.get("message") or payload.get("log")
            with app.state.lock:
                if message:
                    job["logs"].append(str(message))
                    job["logs"] = job["logs"][-300:]
                for key in ("epoch", "epochs", "loss", "val_accuracy", "accuracy"):
                    if key in payload:
                        job[key] = payload[key]
                if "progress" in payload:
                    job["progress"] = min(100, max(0, float(payload["progress"])))
                elif payload.get("epoch") and payload.get("epochs"):
                    job["progress"] = min(99, 100 * payload["epoch"] / payload["epochs"])

        def run():
            with app.state.lock:
                job["status"] = "running"
            try:
                result = operation(progress)
                with app.state.lock:
                    job.update(status="completed", progress=100, result=result)
                    app.state.predictor_key = None
            except Exception as exc:
                with app.state.lock:
                    job.update(status="failed", error=str(exc))
                    job["logs"].append(str(exc))
            finally:
                with app.state.lock:
                    job["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=run, daemon=True, name=f"faceid-{kind}").start()
        return current_job()

    @app.get("/api/status")
    def status():
        model = checkpoint()
        return {"app": "Face Recognition Studio", "people": PEOPLE,
                "sessions": 6, "checkpoint": model.name if model else None,
                "model_ready": model is not None,
                "threshold_ready": threshold_ready(model),
                "splits": config.get("splits", {"train": [1, 2, 3, 4], "val": [5], "test": [6]}),
                "dataset_exists": data_dir.is_dir(), "job": current_job(),
                "download_ready": bool(os.environ.get("FACEID_DOWNLOAD_ARCHIVE")
                    and Path(os.environ["FACEID_DOWNLOAD_ARCHIVE"]).is_file())}

    @app.get("/api/dataset")
    def dataset():
        return service("audit", "faceid.data", "audit_dataset")(config, strict=False)

    @app.post("/api/dataset/init")
    def initialize():
        service("initialize_dataset", "faceid.data", "initialize_dataset")(data_dir)
        return {"ok": True, "message": "Đã tạo cấu trúc 3 người × 6 phiên."}

    @app.post("/api/predict")
    async def predict(image: UploadFile = File(...)):
        _, decoded, _ = await read_image(image)
        predictor = get_predictor()
        def run():
            with app.state.inference_lock:
                return predictor.predict(decoded)
        return await run_in_threadpool(run)

    @app.post("/api/capture", status_code=201)
    async def capture(person: str = Form(...), session: int = Form(...),
                      image: UploadFile = File(...)):
        if person not in PEOPLE or not 1 <= session <= 6:
            raise HTTPException(400, "Chọn Dũng, Duy hoặc Hưng và phiên từ 1 đến 6.")
        raw, decoded, extension = await read_image(image)
        detector = overrides.get("detector")
        if detector is None:
            detector = service("FaceDetector", "faceid.preprocess", "FaceDetector")()
        faces = detector.detect(decoded)
        if len(faces) != 1:
            raise HTTPException(400,
                f"Cần đúng một khuôn mặt để lưu dữ liệu; hiện phát hiện {len(faces)} khuôn mặt.")
        folder = data_dir / person / f"phien_{session:02d}"
        if not folder.resolve().is_relative_to(data_dir.resolve()):
            raise HTTPException(400, "Thư mục lưu ảnh không hợp lệ.")
        # Person and session are validated enumerations; names never come from uploads.
        folder.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12] + extension
        destination = folder / filename
        with destination.open("xb") as stream:
            stream.write(raw)  # raw original image, no drawn boxes or double crop
        return {"ok": True, "person": person, "display_name": PEOPLE[person],
                "session": session, "filename": filename,
                "path": destination.relative_to(data_dir).as_posix(),
                "message": f"Đã lưu ảnh của {PEOPLE[person]} vào phiên {session:02d}."}

    @app.post("/api/train", status_code=202)
    def train(request: TrainRequest):
        if request.backbone not in {"resnet18", "facecnn"}:
            raise HTTPException(400, "Kiến trúc hợp lệ: resnet18 hoặc facecnn.")
        train_config = copy.deepcopy(config)
        train_config.setdefault("training", {}).update(
            epochs=request.epochs, batch_size=request.batch_size,
            backbone=request.backbone, pretrained=request.backbone == "resnet18")
        return start_job("train", lambda progress: service(
            "train", "faceid.engine", "train")(train_config, progress=progress))

    @app.post("/api/evaluate", status_code=202)
    def evaluate():
        selected = checkpoint()
        if selected is None:
            raise HTTPException(409, "Chưa có mô hình để đánh giá.")
        evaluation_config = {**copy.deepcopy(config), "checkpoint": str(selected)}
        return start_job("evaluate", lambda progress: service(
            "evaluate", "faceid.engine", "evaluate")(evaluation_config, split="test"))

    @app.post("/api/legacy/import", status_code=202)
    def import_legacy():
        return start_job("import", lambda progress: service(
            "import_legacy", "faceid.engine", "import_legacy")(config))

    @app.post("/api/calibrate", status_code=202)
    def calibrate(request: CalibrationRequest):
        selected = checkpoint()
        if selected is None:
            raise HTTPException(409, "Chưa có mô hình để hiệu chỉnh ngưỡng.")
        calibration_config = {**copy.deepcopy(config), "checkpoint": str(selected)}
        # Trusted local configuration only; the API never accepts a filesystem path.
        calibration = config.get("calibration", {})
        unknown = Path(calibration.get("unknown_dir") or data_dir.parent / "unknown_val")
        if not unknown.is_absolute():
            base = Path(config.get("config_path") or data_dir.parent / "config.json").parent
            unknown = (base / unknown).resolve()
        if not unknown.is_dir():
            raise HTTPException(400, "Chưa có bộ ảnh người lạ để hiệu chỉnh. Hãy thêm ảnh vào data/unknown_val.")
        return start_job("calibrate", lambda progress: service(
            "calibrate", "faceid.calibration", "calibrate")(
                calibration_config, unknown_dir=unknown, max_far=request.max_far))

    @app.get("/api/jobs/current")
    def job():
        return {"job": current_job()}

    @app.get("/api/reports")
    def reports():
        result = {}
        candidates = {
            "history": ("history.json", "training_history.json", "train_history.json"),
            "validation": ("validation.json", "validation_metrics.json", "val_metrics.json"),
            "test": ("test.json", "test_metrics.json", "evaluation.json"),
            "calibration": ("calibration.json", "threshold.json", "thresholds.json"),
            "summary": ("summary.json", "training_summary.json", "legacy_import.json"),
        }
        for key, names in candidates.items():
            for name in names:
                path = artifacts_dir / name
                if path.is_file() and path.stat().st_size <= 5 * 1024 * 1024:
                    try:
                        result[key] = json.loads(path.read_text(encoding="utf-8"))
                        break
                    except (OSError, ValueError):
                        continue
        return {"reports": result, "available": bool(result)}

    @app.get("/api/download-project")
    def download_project():
        configured = os.environ.get("FACEID_DOWNLOAD_ARCHIVE")
        archive = Path(configured) if configured else None
        if archive is None or not archive.is_file() or archive.suffix != ".zip":
            raise HTTPException(404, "Bản tải xuống chưa được chuẩn bị.")
        return FileResponse(archive, media_type="application/zip",
                            filename="face-recognition-studio.zip")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
