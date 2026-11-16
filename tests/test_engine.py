"""Synthetic end-to-end checks; no external face photos or weight downloads."""
import copy
import json
import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from faceid.engine import evaluate, import_legacy, train, score_loader
from faceid.calibration import calibrate
from faceid.inference import Predictor
from faceid.models import ArcFace, FaceCNN


CLASSES = ["dung", "duy", "hung"]


def make_config(root, epochs=1):
    root = Path(root)
    return {
        "input_mode": "raw_image",
        "data_dir": str(root / "people"),
        "artifacts_dir": str(root / "artifacts"),
        "legacy_dir": str(Path(__file__).resolve().parents[1] / "weights" / "legacy"),
        "splits": {
            "train": [1, 2, 3, 4],
            "val": [5],
            "test": [6],
        },
        "preprocess": {
            "size": [32, 32], "resize_mode": "pad", "normalization": "none",
            "crop_face": False, "face_margin": 0.2, "no_face": "error",
        },
        "training": {
            "backbone": "tinycnn", "pretrained": False, "embedding_dim": 8,
            "epochs": epochs, "batch_size": 12, "lr": 0.001,
            "weight_decay": 0.0001, "patience": 2, "seed": 42,
            "num_workers": 0, "device": "cpu",
        },
    }


def populate(config):
    for label, person in enumerate(CLASSES):
        for session in range(1, 7):
            directory = Path(config["data_dir"]) / person / f"phien_{session:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                rng = np.random.default_rng(label * 100 + session * 10 + index)
                pixels = rng.integers(0, 100, (32, 32, 3), dtype=np.uint8)
                pixels[:, :, label] += 120
                Image.fromarray(pixels).save(directory / f"{index:03d}.png")


class ModelTests(unittest.TestCase):
    def test_arcface_gradients_finite_and_no_margin_in_eval(self):
        head = ArcFace(embedding_dim=8, num_classes=3)
        embeddings = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0])
        head.train()
        logits = head(embeddings, labels)
        self.assertEqual(tuple(logits.shape), (4, 3))
        F.cross_entropy(logits, labels).backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all().item())
        self.assertTrue(torch.isfinite(head.weight.grad).all().item())
        head.eval()
        without_labels = head(embeddings.detach())
        with_labels = head(embeddings.detach(), labels)
        torch.testing.assert_close(with_labels, without_labels)
        expected = F.linear(F.normalize(embeddings.detach(), dim=1),
                            F.normalize(head.weight, dim=1)) * head.scale
        torch.testing.assert_close(without_labels, expected, rtol=1e-5, atol=1e-5)

    def test_arcface_parallel_vectors_have_finite_loss_and_gradients(self):
        head = ArcFace(embedding_dim=8, num_classes=3)
        with torch.no_grad():
            head.weight.copy_(torch.eye(3, 8))
        embeddings = torch.eye(3, 8).clone().requires_grad_(True)
        loss = F.cross_entropy(head(embeddings, torch.arange(3)), torch.arange(3))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(embeddings.grad).all().item())
        self.assertTrue(torch.isfinite(head.weight.grad).all().item())

    def test_uploaded_legacy_backbone_loads_exactly(self):
        path = Path(__file__).resolve().parents[1] / "weights" / "legacy" / "arcface_backbone.pth"
        model = FaceCNN(embedding_dim=128)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.assertEqual(set(state), set(model.state_dict()))
        model.eval()
        with torch.inference_mode():
            output = model(torch.rand(2, 3, 112, 112))
        self.assertEqual(tuple(output.shape), (2, 128))
        self.assertTrue(torch.isfinite(output).all().item())


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = make_config(cls.temp.name)
        populate(cls.config)
        cls.summary = train(copy.deepcopy(cls.config))
        cls.checkpoint = Path(cls.config["artifacts_dir"]) / "best.pth"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_train_writes_self_describing_safe_bundle(self):
        json.dumps(self.summary, ensure_ascii=False, allow_nan=False)
        self.assertTrue(self.checkpoint.is_file())
        bundle = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(bundle["classes"], CLASSES)
        self.assertEqual(bundle["architecture"], "tinycnn")
        self.assertEqual(bundle["embedding_dim"], 8)
        for key, value in self.config["preprocess"].items():
            self.assertEqual(bundle["preprocess"][key], value)
        self.assertEqual(tuple(bundle["prototypes"].shape), (3, 8))
        torch.testing.assert_close(bundle["prototypes"].norm(dim=1), torch.ones(3))
        self.assertTrue(bundle["train_hashes"])
        self.assertIn("val", bundle["split_signatures"])
        self.assertIn("preprocess_id", bundle)
        self.assertIn("metrics", bundle)
        self.assertGreaterEqual(bundle["epoch"], 1)

    def test_evaluate_is_json_serializable_and_does_not_change_checkpoint(self):
        before = self.checkpoint.read_bytes()
        report = evaluate(copy.deepcopy(self.config), split="test")
        json.dumps(report, ensure_ascii=False, allow_nan=False)
        self.assertEqual(report["split"], "test")
        self.assertEqual(self.checkpoint.read_bytes(), before)
        matrix = report.get("confusion_matrix", report.get("confusion"))
        self.assertIsNotNone(matrix)
        self.assertEqual(np.asarray(matrix).shape, (3, 3))
        self.assertEqual(np.asarray(matrix).sum(), 6)
        self.assertGreaterEqual(report["macro_f1"], 0)
        self.assertLessEqual(report["macro_f1"], 1)

    def test_evaluation_rejects_dataset_content_change(self):
        path = Path(self.config["data_dir"]) / "dung" / "phien_06" / "000.png"
        original = path.read_bytes()
        try:
            Image.new("RGB", (32, 32), (11, 22, 33)).save(path)
            with self.assertRaises(ValueError):
                evaluate(copy.deepcopy(self.config), split="test")
        finally:
            path.write_bytes(original)

    def test_evaluation_threshold_comparison_preserves_float64_boundary(self):
        bundle = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        threshold_path = Path(self.config["artifacts_dir"]) / "threshold.json"
        try:
            for score in (0.5, 1.0):
                with self.subTest(score=score):
                    threshold = math.nextafter(score, math.inf)
                    threshold_path.write_text(json.dumps({
                        "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                        "preprocess_id": bundle["preprocess_id"], "classes": CLASSES,
                        "threshold": threshold,
                    }), encoding="utf-8")
                    scores = torch.eye(3, dtype=torch.float32) * score
                    with patch("faceid.engine.score_loader", return_value=(scores, torch.arange(3))):
                        report = evaluate(copy.deepcopy(self.config), split="test")
                    self.assertEqual(report["open_set"]["accepted"], 0)
                    self.assertEqual(report["open_set"]["rejected"], 3)
        finally:
            threshold_path.unlink(missing_ok=True)

    def test_calibration_scores_only_validation_and_unknown_and_binds_threshold(self):
        unknown = Path(self.temp.name) / "unknown_validation"
        unknown.mkdir(exist_ok=True)
        for index in range(4):
            pixels = np.random.default_rng(900 + index).integers(
                0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(unknown / f"{index}.png")
        observed = []
        def record_scores(model, centers, loader, device):
            observed.extend(path for path, _ in loader.dataset.samples)
            return score_loader(model, centers, loader, device)
        threshold_path = Path(self.config["artifacts_dir"]) / "threshold.json"
        try:
            with patch("faceid.calibration.score_loader", side_effect=record_scores):
                report = calibrate(copy.deepcopy(self.config), unknown, max_far=0.25)
            json.dumps(report, ensure_ascii=False, allow_nan=False)
            self.assertFalse(report["test_used_for_threshold"])
            self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
            self.assertTrue(threshold_path.is_file())
            self.assertEqual(report["unknown_count"], 4)
            self.assertLessEqual(report["empirical_unknown_far"], 0.25)
            self.assertTrue(observed)
            self.assertTrue(all("phien_06" not in Path(path).parts for path in observed))
            self.assertTrue(all("phien_0" not in str(path) or "phien_05" in str(path) for path in observed))
            predictor = Predictor(self.checkpoint, threshold_file=threshold_path, device="cpu")
            self.assertTrue(predictor.metadata["calibrated"])
            evaluation = evaluate(copy.deepcopy(self.config), split="test")
            self.assertIsNotNone(evaluation["open_set"])
            self.assertEqual(evaluation["open_set"]["threshold"], report["threshold"])
            json.dumps(evaluation, allow_nan=False)
        finally:
            threshold_path.unlink(missing_ok=True)

    def test_calibration_rejects_unknown_copy_of_known_data(self):
        unknown = Path(self.temp.name) / "copied_unknown"
        unknown.mkdir(exist_ok=True)
        known = Path(self.config["data_dir"]) / "duy" / "phien_06" / "000.png"
        (unknown / "copy.png").write_bytes(known.read_bytes())
        threshold_path = Path(self.config["artifacts_dir"]) / "threshold.json"
        bundle = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        threshold_path.write_text(json.dumps({
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "preprocess_id": bundle["preprocess_id"], "classes": CLASSES,
            "threshold": 0.8,
        }), encoding="utf-8")
        previous = threshold_path.read_bytes()
        try:
            with self.assertRaises(ValueError):
                calibrate(copy.deepcopy(self.config), unknown)
            self.assertEqual(threshold_path.read_bytes(), previous)
        finally:
            threshold_path.unlink(missing_ok=True)

    def test_second_training_archives_stale_checkpoint_threshold_and_test_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            config = make_config(temp)
            populate(config)
            first = train(copy.deepcopy(config))
            artifacts = Path(config["artifacts_dir"])
            previous_bytes = (artifacts / "best.pth").read_bytes()
            (artifacts / "threshold.json").write_text('{"old": true}', encoding="utf-8")
            (artifacts / "test_metrics.json").write_text('{"old": true}', encoding="utf-8")
            second = train(copy.deepcopy(config))
            archive = Path(second["archived_previous_run"])
            self.assertTrue(archive.is_dir())
            self.assertEqual((archive / "best.pth").read_bytes(), previous_bytes)
            self.assertEqual(json.loads((archive / "threshold.json").read_text()), {"old": True})
            self.assertTrue((archive / "test_metrics.json").is_file())
            self.assertTrue((archive / "history.json").is_file())
            self.assertTrue((artifacts / "best.pth").is_file())
            self.assertFalse((artifacts / "threshold.json").exists())
            self.assertFalse((artifacts / "test_metrics.json").exists())

    def test_import_legacy_marks_unvalidated_and_preserves_class_order(self):
        report = import_legacy(copy.deepcopy(self.config))
        json.dumps(report, ensure_ascii=False, allow_nan=False)
        path = Path(self.config["artifacts_dir"]) / "legacy.pth"
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(bundle["classes"], CLASSES)
        self.assertEqual(bundle["architecture"], "facecnn")
        self.assertEqual(bundle["embedding_dim"], 128)
        torch.testing.assert_close(bundle["prototypes"].norm(dim=1), torch.ones(3))
        self.assertEqual(bundle["preprocess"]["size"], [112, 112])
        self.assertEqual(bundle["preprocess"]["resize_mode"], "resize")
        self.assertEqual(bundle["preprocess"]["normalization"], "none")
        self.assertFalse(bundle["preprocess"]["crop_face"])


if __name__ == "__main__":
    unittest.main()
