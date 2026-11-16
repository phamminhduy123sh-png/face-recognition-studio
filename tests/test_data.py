"""Dataset audit and session-isolation checks with synthetic images only."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from faceid.data import CLASSES, DatasetError, FaceDataset, audit_dataset, initialize_dataset, load_splits


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        initialize_dataset(self.root)
        self.config = {
            "data_dir": str(self.root),
            "splits": {"train": [1, 2, 3, 4], "val": [5], "test": [6]},
            "preprocess": {"size": [16, 16], "crop_face": False},
        }

    def tearDown(self):
        self.temp.cleanup()

    def populate(self):
        rng = np.random.default_rng(11)
        for person in CLASSES:
            for session in range(1, 7):
                array = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
                Image.fromarray(array).save(self.root / person / f"phien_{session:02d}" / "photo.png")

    def test_initialize_all_eighteen_without_overwriting(self):
        marker = self.root / "dung" / "phien_01" / "notes.txt"
        marker.write_text("keep", encoding="utf-8")
        folders = initialize_dataset(self.root)
        self.assertEqual(len(folders), 18)
        self.assertTrue(all(Path(folder).is_dir() for folder in folders))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        report = audit_dataset(self.config)
        self.assertFalse(report["valid"])
        self.assertEqual(sum(issue["code"] == "empty_session" for issue in report["issues"]), 18)
        self.assertEqual(report["counts"]["by_person_session"]["dung"]["phien_01"], 0)
        json.dumps(report)

    def test_valid_sessions_fixed_labels_and_disjoint_splits(self):
        self.populate()
        report = audit_dataset(self.config)
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["class_to_idx"], {"dung": 0, "duy": 1, "hung": 2})
        self.assertEqual(report["counts"]["by_split"], {"train": 12, "val": 3, "test": 3})
        splits = load_splits(self.config)
        sets = {name: {path for path, _ in samples} for name, samples in splits.items()}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        self.assertTrue(all("/phien_05/" in path for path, _ in splits["val"]))
        self.assertTrue(all("/phien_06/" in path for path, _ in splits["test"]))
        self.assertEqual({label for _, label in splits["test"]}, {0, 1, 2})

    def test_portable_manifest_signature_changes_with_image(self):
        self.populate()
        original = audit_dataset(self.config)
        clone = Path(self.temp.name) / "clone"
        shutil.copytree(self.root, clone)
        copied_config = {**self.config, "data_dir": str(clone)}
        copied = audit_dataset(copied_config)
        self.assertEqual(original["signature"], copied["signature"])
        self.assertTrue(all(not Path(row["path"]).is_absolute() for row in original["manifest"]))
        Image.new("RGB", (25, 25), "purple").save(clone / "dung" / "phien_01" / "photo.png")
        self.assertNotEqual(original["signature"], audit_dataset(copied_config)["signature"])

    def test_exact_duplicate_across_splits_is_error(self):
        self.populate()
        source = self.root / "dung" / "phien_01" / "photo.png"
        destination = self.root / "dung" / "phien_05" / "photo.png"
        shutil.copyfile(source, destination)
        report = audit_dataset(self.config)
        self.assertFalse(report["valid"])
        self.assertTrue(any(item["code"] == "duplicate_leakage" for item in report["issues"]))
        with self.assertRaises(DatasetError):
            load_splits(self.config)

    def test_pixel_duplicate_in_different_encoding_across_persons_is_error(self):
        self.populate()
        source = self.root / "dung" / "phien_01" / "photo.png"
        destination = self.root / "duy" / "phien_01" / "copy.bmp"
        with Image.open(source) as image:
            image.save(destination)
        report = audit_dataset(self.config)
        self.assertFalse(report["valid"])
        self.assertTrue(any(item["kind"] == "pixels" and item["severity"] == "error" for item in report["duplicates"]))

    def test_same_person_training_duplicate_only_warns(self):
        self.populate()
        source = self.root / "dung" / "phien_01" / "photo.png"
        shutil.copyfile(source, self.root / "dung" / "phien_02" / "copy.png")
        report = audit_dataset(self.config)
        self.assertTrue(report["valid"], report["issues"])
        self.assertTrue(any(item["code"] == "duplicate_image" and item["severity"] == "warning" for item in report["issues"]))

    def test_corrupt_file_and_missing_session_are_reported(self):
        self.populate()
        (self.root / "hung" / "phien_02" / "bad.jpg").write_bytes(b"not an image")
        shutil.rmtree(self.root / "duy" / "phien_06")
        report = audit_dataset(self.config)
        self.assertFalse(report["valid"])
        codes = {item["code"] for item in report["issues"]}
        self.assertTrue({"corrupt_image", "missing_session"}.issubset(codes))

    def test_alias_session_requires_explicit_migration(self):
        self.populate()
        folder = self.root / "dung" / "phien_01"
        folder.rename(folder.with_name("session_01"))
        report = audit_dataset(self.config)
        self.assertFalse(report["valid"])
        self.assertTrue(any(item["code"] == "unexpected_session" for item in report["issues"]))
        self.assertFalse(folder.exists())
        self.assertTrue(folder.with_name("session_01").is_dir())

    def test_overlapping_sessions_never_load(self):
        self.populate()
        bad = {**self.config, "splits": {"train": [1, 2, 3, 4, 5], "val": [5], "test": [6]}}
        report = audit_dataset(bad)
        self.assertFalse(report["valid"])
        self.assertTrue(any(item["code"] == "configuration" for item in report["issues"]))
        with self.assertRaises(DatasetError):
            load_splits(bad)

    def test_strict_face_audit_reports_failures_quick_audit_does_not_detect(self):
        self.populate()
        config = {**self.config, "preprocess": {"size": [16, 16], "crop_face": True}}
        with patch("faceid.data.FacePreprocessor") as constructor:
            constructor.return_value.prepare.side_effect = ValueError("Multiple faces detected")
            strict_report = audit_dataset(config, strict=True)
            self.assertEqual(constructor.return_value.prepare.call_count, 18)
            self.assertFalse(strict_report["valid"])
            self.assertEqual(sum(issue["code"] == "face_validation" for issue in strict_report["issues"]), 18)
        with patch("faceid.data.FacePreprocessor") as constructor:
            quick_report = audit_dataset(config, strict=False)
            constructor.assert_not_called()
            self.assertTrue(quick_report["valid"], quick_report["issues"])

    def test_dataset_outputs_tensor_and_fixed_integer_label(self):
        self.populate()
        samples = load_splits(self.config)["train"]
        dataset = FaceDataset(samples, self.config["preprocess"], training=False)
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(image.dtype, torch.float32)
        self.assertIsInstance(label, int)
        self.assertEqual(label, samples[0][1])
        torch.testing.assert_close(image, dataset[0][0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
