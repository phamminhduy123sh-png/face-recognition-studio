"""Local web API tests with generated images and injected model services."""
import io
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from faceid.web.app import create_app


class FakeDetector:
    def __init__(self, boxes=None):
        self.boxes = [(1, 1, 10, 10)] if boxes is None else boxes

    def detect(self, image):
        return self.boxes


class FakePredictor:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, image):
        return {
            "width": image.width,
            "height": image.height,
            "faces": [{
                "bbox": [1, 1, 10, 10],
                "person": "duy",
                "display_name": "Duy",
                "score": 0.72,
                "accepted": False,
                "status": "uncalibrated",
            }],
        }


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {
            "data_dir": str(self.root / "dataset"),
            "artifacts_dir": str(self.root / "artifacts"),
            "legacy_dir": str(self.root / "legacy"),
            "device": "cpu",
            "model": {"backbone": "ir18", "embedding_dim": 32},
            "training": {"epochs": 2, "batch_size": 2},
            "preprocess": {"size": [112, 112]},
        }
        self.detector = FakeDetector()
        self.services = {
            "config": self.config,
            "detector": self.detector,
            "audit": lambda config, strict=False: {
                "valid": False,
                "counts": {},
                "issues": ["Dataset chưa có ảnh"],
            },
            "Predictor": FakePredictor,
        }
        self.client = TestClient(create_app(services=self.services))

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    @staticmethod
    def image_bytes():
        # Distinct pixels reveal accidental detector boxes or face cropping.
        pixels = np.zeros((20, 30, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(30, dtype=np.uint8)[None, :] * 8
        pixels[:, :, 1] = np.arange(20, dtype=np.uint8)[:, None] * 10
        pixels[:, :, 2] = 125
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        return stream.getvalue()

    def upload(self, endpoint="/api/predict", image=None, **data):
        if image is None:
            image = self.image_bytes()
        return self.client.post(
            endpoint,
            data=data,
            files={"image": ("photo.png", image, "image/png")},
        )

    def test_home_and_status_work_without_dataset_or_checkpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)
        self.assertNotIn("traceback", response.text.lower())

    def test_empty_dataset_and_reports_are_json_responses(self):
        for endpoint in ("/api/dataset", "/api/reports", "/api/jobs/current"):
            with self.subTest(endpoint=endpoint):
                response = self.client.get(endpoint)
                self.assertEqual(response.status_code, 200)
                self.assertIsInstance(response.json(), (dict, list))

    def test_init_creates_exactly_eighteen_session_folders(self):
        response = self.client.post("/api/dataset/init")
        self.assertIn(response.status_code, (200, 201))
        root = Path(self.config["data_dir"])
        actual = {p.relative_to(root).as_posix()
                  for p in root.glob("*/*") if p.is_dir()}
        expected = {f"{person}/phien_{session:02d}"
                    for person in ("dung", "duy", "hung")
                    for session in range(1, 7)}
        self.assertEqual(actual, expected)

    def test_predict_requires_model_instead_of_inventing_result(self):
        response = self.upload()
        self.assertIn(response.status_code, (400, 404, 409, 503))
        self.assertNotIn("traceback", response.text.lower())

    def test_prediction_keeps_uncalibrated_candidate_unaccepted(self):
        checkpoint = Path(self.config["artifacts_dir"]) / "best.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"injected fake predictor does not load weights")
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result["faces"]), 1)
        candidate = result["faces"][0]
        self.assertEqual(candidate["person"], "duy")
        self.assertEqual(candidate["status"], "uncalibrated")
        self.assertFalse(candidate["accepted"])
        self.assertEqual(candidate["score"], 0.72)

    def test_predict_composites_transparent_image_to_white_rgb(self):
        checkpoint = Path(self.config["artifacts_dir"]) / "best.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"injected fake weights")
        observed = []

        class PixelCheckingPredictor(FakePredictor):
            def predict(self, image):
                observed.append((image.mode, np.asarray(image).copy()))
                return {"width": image.width, "height": image.height, "faces": []}

        stream = io.BytesIO()
        Image.new("RGBA", (12, 10), (0, 0, 0, 0)).save(stream, format="PNG")
        services = {**self.services, "Predictor": PixelCheckingPredictor}
        with TestClient(create_app(services=services)) as client:
            response = client.post("/api/predict", files={
                "image": ("transparent.png", stream.getvalue(), "image/png")
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], "RGB")
        np.testing.assert_array_equal(
            observed[0][1], np.full((10, 12, 3), 255, dtype=np.uint8))

    def test_capture_saves_original_full_frame_and_unique_names(self):
        original = self.image_bytes()
        first = self.upload("/api/capture", image=original, person="duy", session="3")
        second = self.upload("/api/capture", image=original, person="duy", session="3")
        self.assertIn(first.status_code, (200, 201), first.text)
        self.assertIn(second.status_code, (200, 201), second.text)
        folder = Path(self.config["data_dir"]) / "duy" / "phien_03"
        files = [p for p in folder.iterdir() if p.is_file()]
        self.assertEqual(len(files), 2)
        expected = np.asarray(Image.open(io.BytesIO(original)).convert("RGB"))
        for path in files:
            self.assertEqual(path.read_bytes(), original)
            with Image.open(path) as image:
                self.assertEqual(image.size, (30, 20))
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), expected)
        # Never leak the service's private absolute data path to the browser.
        self.assertNotIn(str(self.root), first.text)

    def test_capture_rejects_invalid_person_session_and_traversal(self):
        invalid = [
            {"person": "other", "session": "1"},
            {"person": "../duy", "session": "1"},
            {"person": "duy", "session": "0"},
            {"person": "duy", "session": "7"},
            {"person": "duy", "session": "../../outside"},
        ]
        for data in invalid:
            with self.subTest(data=data):
                response = self.upload("/api/capture", **data)
                self.assertIn(response.status_code, (400, 422))
        self.assertFalse(any(Path(self.config["data_dir"]).rglob("*.png")))

    def test_capture_requires_exactly_one_face(self):
        for boxes in ([], [(0, 0, 8, 8), (10, 0, 8, 8)]):
            with self.subTest(boxes=boxes):
                self.detector.boxes = boxes
                response = self.upload("/api/capture", person="dung", session="1")
                self.assertIn(response.status_code, (400, 422))
        self.assertFalse(any(Path(self.config["data_dir"]).rglob("*.png")))

    def test_capture_loads_real_detector_service_and_rejects_no_face(self):
        services = {key: value for key, value in self.services.items()
                    if key != "detector"}
        with TestClient(create_app(services=services)) as client:
            response = client.post("/api/capture", data={
                "person": "dung", "session": "1"
            }, files={"image": ("no-face.png", self.image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("khuôn mặt", response.json()["detail"])
        self.assertFalse(any(Path(self.config["data_dir"]).rglob("*.png")))

    def test_capture_rejects_corrupt_empty_and_oversized_uploads(self):
        for value in (b"not an image", b"", b"x" * (10 * 1024 * 1024 + 1)):
            with self.subTest(length=len(value)):
                response = self.upload("/api/capture", image=value, person="dung", session="1")
                self.assertIn(response.status_code, (400, 413, 422))
                self.assertNotIn("traceback", response.text.lower())

    def test_training_allows_only_one_running_job(self):
        entered = threading.Event()
        release = threading.Event()

        def train(config, progress=None):
            entered.set()
            release.wait(timeout=3)
            return {"completed": True}

        services = {**self.services, "train": train}
        client = TestClient(create_app(services=services))
        try:
            response = client.post("/api/train", json={
                "epochs": 2, "backbone": "facecnn", "batch_size": 2,
            })
            self.assertIn(response.status_code, (200, 202), response.text)
            self.assertTrue(entered.wait(timeout=2))
            second = client.post("/api/train", json={
                "epochs": 2, "backbone": "facecnn", "batch_size": 2,
            })
            self.assertEqual(second.status_code, 409, second.text)
        finally:
            release.set()
            client.close()

    def test_calibration_requires_default_unknown_validation_directory(self):
        checkpoint = Path(self.config["artifacts_dir"]) / "legacy.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"injected fake legacy weights")
        response = self.client.post("/api/calibrate", json={"max_far": 0.05})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("người lạ", response.json()["detail"])
        self.assertIsNone(self.client.get("/api/jobs/current").json()["job"])

    def test_calibration_uses_fixed_default_directory_not_request_path(self):
        checkpoint = Path(self.config["artifacts_dir"]) / "legacy.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"injected fake legacy weights")
        expected = Path(self.config["data_dir"]).parent / "unknown_val"
        expected.mkdir()
        called = threading.Event()
        observed = []

        def calibrate(config, unknown_dir, max_far):
            observed.append((Path(unknown_dir), max_far, config.get("checkpoint")))
            called.set()
            return {"threshold": 0.5}

        services = {**self.services, "calibrate": calibrate}
        with TestClient(create_app(services=services)) as client:
            response = client.post("/api/calibrate", json={
                "max_far": 0.02,
                "unknown_dir": "/arbitrary/request/directory",
            })
            self.assertEqual(response.status_code, 202, response.text)
            self.assertTrue(called.wait(timeout=2))
        self.assertEqual(observed, [(expected, 0.02, str(checkpoint))])

    def test_evaluation_passes_active_legacy_checkpoint_to_engine(self):
        checkpoint = Path(self.config["artifacts_dir"]) / "legacy.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"injected fake legacy weights")
        called = threading.Event()
        observed = []

        def evaluate(config, split="test"):
            observed.append((config.get("checkpoint"), split))
            called.set()
            return {"accuracy": 0.5}

        services = {**self.services, "evaluate": evaluate}
        with TestClient(create_app(services=services)) as client:
            response = client.post("/api/evaluate")
            self.assertEqual(response.status_code, 202, response.text)
            self.assertTrue(called.wait(timeout=2))
        self.assertEqual(observed, [(str(checkpoint), "test")])

    def test_evaluation_and_calibration_require_checkpoint_before_starting_job(self):
        for endpoint, payload in (("/api/evaluate", None),
                                  ("/api/calibrate", {"max_far": 0.05})):
            with self.subTest(endpoint=endpoint):
                response = self.client.post(endpoint, json=payload)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIsNone(self.client.get("/api/jobs/current").json()["job"])


if __name__ == "__main__":
    unittest.main()
