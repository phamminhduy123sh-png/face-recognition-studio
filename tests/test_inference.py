"""Check safe unknown decisions, checkpoint binding and per-face crops."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

from faceid.inference import Predictor
from faceid.calibration import choose_threshold
from faceid.models import build_backbone
from faceid.preprocess import FaceDetector, preprocess_id


CLASSES = ["dung", "duy", "hung"]


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "tiny.pth"
        self.preprocess = dict(size=[32, 32], resize_mode="pad", normalization="none",
                               crop_face=False, face_margin=0.2, no_face="error")
        model = build_backbone("tinycnn", embedding_dim=8, pretrained=False)
        torch.manual_seed(1)
        self.bundle = {
            "format": "face-recognition-studio", "version": 1,
            "architecture": "tinycnn", "embedding_dim": 8, "classes": CLASSES,
            "preprocess": self.preprocess, "preprocess_id": preprocess_id(self.preprocess),
            "backbone": model.state_dict(), "head": {},
            "prototypes": F.normalize(torch.randn(3, 8), dim=1),
            "splits": {}, "metrics": {}, "epoch": 0,
            "train_hashes": [], "split_signatures": {}, "data_signature": "synthetic",
        }
        torch.save(self.bundle, self.checkpoint)
        self.image = Image.new("RGB", (80, 64), (100, 80, 60))

    def tearDown(self):
        self.temp.cleanup()

    def threshold_file(self, **updates):
        value = {
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "preprocess_id": self.bundle["preprocess_id"], "classes": CLASSES,
            "threshold": 0.8,
        }
        value.update(updates)
        path = self.root / "threshold.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_no_threshold_returns_candidate_without_accepted_identity(self):
        predictor = Predictor(self.checkpoint, device="cpu")
        result = predictor.predict_face(self.image)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertEqual(result["status"], "uncalibrated")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["calibrated"])
        self.assertIn(result["candidate"], CLASSES)
        self.assertIsNone(result["person"])
        self.assertGreaterEqual(result["score"], -1)
        self.assertLessEqual(result["score"], 1)

    def test_predict_face_skips_second_face_detection(self):
        predictor = Predictor(self.checkpoint, device="cpu")
        with patch.object(FaceDetector, "detect", side_effect=AssertionError("double crop")):
            result = predictor.predict_face(self.image)
        self.assertIn(result["candidate"], CLASSES)

    def test_predict_processes_all_detected_faces(self):
        predictor = Predictor(self.checkpoint, device="cpu")
        boxes = [(0, 0, 30, 30), (40, 20, 30, 30)]
        with patch.object(FaceDetector, "detect", return_value=boxes) as detector:
            report = predictor.predict(self.image)
        self.assertEqual(detector.call_count, 1)
        self.assertEqual(len(report["faces"]), 2)
        self.assertEqual([face["bbox"] for face in report["faces"]], [list(box) for box in boxes])
        self.assertTrue(all(not face["accepted"] for face in report["faces"]))

    def test_raw_image_mode_explicitly_bypasses_detector(self):
        self.bundle["input_mode"] = "raw_image"
        torch.save(self.bundle, self.checkpoint)
        predictor = Predictor(self.checkpoint, device="cpu")
        with patch.object(FaceDetector, "detect", side_effect=AssertionError("unexpected detector")):
            report = predictor.predict(self.image)
        self.assertEqual(report["input_mode"], "raw_image")
        self.assertEqual(len(report["faces"]), 1)
        self.assertEqual(report["faces"][0]["bbox"], [0, 0, 80, 64])

    def test_predict_no_faces_returns_empty_report(self):
        predictor = Predictor(self.checkpoint, device="cpu")
        with patch.object(FaceDetector, "detect", return_value=[]):
            report = predictor.predict(self.image)
        self.assertEqual(report["faces"], [])

    def test_threshold_rejects_wrong_checkpoint_or_preprocessing(self):
        for updates in ({"checkpoint_sha256": "0" * 64}, {"preprocess_id": "different"}):
            with self.subTest(updates=updates):
                path = self.threshold_file(**updates)
                with self.assertRaises(ValueError):
                    Predictor(self.checkpoint, threshold_file=path, device="cpu")

    def test_threshold_file_rejects_nonfinite_or_out_of_range_value(self):
        for value in (float("nan"), float("inf"), -1.1, 1.1):
            with self.subTest(value=value):
                path = self.threshold_file(threshold=value)
                with self.assertRaises(ValueError):
                    Predictor(self.checkpoint, threshold_file=path, device="cpu")

    def test_bad_or_nonfinite_manual_threshold_rejected(self):
        for value in (float("nan"), float("inf"), -1.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Predictor(self.checkpoint, device="cpu", manual_threshold=value)

    def test_checkpoint_with_inconsistent_prototype_count_is_rejected(self):
        bad = copy.copy(self.bundle)
        bad["prototypes"] = torch.zeros(2, 8)
        torch.save(bad, self.checkpoint)
        with self.assertRaises(ValueError):
            Predictor(self.checkpoint, device="cpu")


class CalibrationSelectionTests(unittest.TestCase):
    def test_far_constraint_preserves_best_correct_known_accepts(self):
        report = choose_threshold([0.95, 0.8, 0.6], [True, True, False],
                                  [0.9, 0.7, 0.4, 0.2], max_far=0.25)
        self.assertLessEqual(report["empirical_unknown_far"], 0.25)
        self.assertEqual(report["unknown_accepted"], 1)
        self.assertEqual(report["known_accepted"], 2)
        self.assertAlmostEqual(report["known_correct_and_accepted_rate"], 2/3)
        self.assertGreater(report["threshold"], 0.7)
        self.assertLessEqual(report["threshold"], 0.8)

    def test_perfect_unknown_cosine_can_be_rejected_with_threshold_above_one(self):
        report = choose_threshold([1.0], [True], [1.0], max_far=0)
        self.assertEqual(report["threshold"], math.nextafter(1.0, math.inf))
        self.assertEqual(report["unknown_accepted"], 0)
        self.assertEqual(report["known_accepted"], 0)
        self.assertEqual(report["known_coverage"], 0)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_invalid_or_empty_calibration_inputs_rejected(self):
        cases = [([float("nan")], [True], [0.1], .05),
                 ([.5], [True], [float("inf")], .05),
                 ([.5], [True], [1.1], .05),
                 ([], [], [.1], .05),
                 ([.5], [], [.1], .05),
                 ([.5], [True], [.1], float("nan")),
                 ([.5], [True], [.1], -.1)]
        for known, correct, unknown, max_far in cases:
            with self.subTest(known=known, unknown=unknown, max_far=max_far):
                with self.assertRaises(ValueError):
                    choose_threshold(known, correct, unknown, max_far)


if __name__ == "__main__":
    unittest.main()
