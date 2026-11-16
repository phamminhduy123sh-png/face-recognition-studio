"""Generated-image tests for shared training/inference preprocessing."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from faceid.preprocess import FaceDetector, FacePreprocessor, preprocess_id


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.boxes

    def crop(self, image, bbox, margin=0.2):
        return FaceDetector().crop(image, bbox, margin=margin)


class PreprocessTests(unittest.TestCase):
    def config(self, **updates):
        config = dict(size=[16, 20], resize_mode="pad", normalization="none",
                      crop_face=False, face_margin=0.2, no_face="error")
        config.update(updates)
        return config

    def test_rgb_tensor_shape_dtype_and_known_pixel(self):
        image = Image.new("RGB", (20, 16), (255, 128, 0))
        tensor = FacePreprocessor(self.config())(image)
        self.assertEqual(tuple(tensor.shape), (3, 16, 20))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all().item())
        torch.testing.assert_close(tensor[:, 8, 10], torch.tensor([1., 128/255, 0.]))

    def test_alpha_is_composited_on_white_and_grayscale_is_rgb(self):
        processor = FacePreprocessor(self.config())
        transparent = Image.new("RGBA", (20, 16), (0, 0, 0, 0))
        torch.testing.assert_close(processor(transparent), torch.ones(3, 16, 20))
        gray = processor(Image.new("L", (20, 16), 128))
        torch.testing.assert_close(gray, torch.full((3, 16, 20), 128/255))

    def test_exif_orientation_is_applied_for_path(self):
        image = Image.new("RGB", (10, 20), "red")
        image.paste("blue", (0, 10, 10, 20))
        exif = Image.Exif()
        exif[274] = 6
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oriented.jpg"
            image.save(path, quality=100, subsampling=0, exif=exif)
            tensor = FacePreprocessor(self.config(size=[10, 20]))(path)
        self.assertGreater(tensor[2, 5, 2].item(), .9)
        self.assertGreater(tensor[0, 5, 17].item(), .9)

    def test_normalization_modes(self):
        image = Image.new("RGB", (20, 16), "white")
        symmetric = FacePreprocessor(self.config(normalization="symmetric"))(image)
        torch.testing.assert_close(symmetric, torch.ones_like(symmetric))
        black = FacePreprocessor(self.config(normalization="symmetric"))(
            Image.new("RGB", (20, 16), "black"))
        torch.testing.assert_close(black, -torch.ones_like(black))
        imagenet = FacePreprocessor(self.config(normalization="imagenet"))(image)
        expected = (torch.ones(3) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(imagenet[:, 8, 10], expected)

    def test_validation_is_deterministic_and_training_augments(self):
        rng = np.random.default_rng(8)
        image = Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8))
        validation = FacePreprocessor(self.config(), training=False)
        torch.testing.assert_close(validation(image), validation(image), rtol=0, atol=0)
        train = FacePreprocessor(self.config(), training=True)
        torch.manual_seed(90)
        outputs = [train(image) for _ in range(4)]
        self.assertTrue(any(not torch.equal(outputs[0], value) for value in outputs[1:]))

    def test_already_cropped_never_detects_a_second_time(self):
        processor = FacePreprocessor(self.config(crop_face=True))
        detector = FakeDetector([])
        processor._detector = detector
        tensor = processor.tensor(Image.new("RGB", (20, 16), "red"), already_cropped=True)
        self.assertEqual(tuple(tensor.shape), (3, 16, 20))
        self.assertEqual(detector.calls, 0)

    def test_crop_disabled_preserves_whole_image_without_detection(self):
        processor = FacePreprocessor(self.config(crop_face=False, resize_mode="resize"))
        detector = FakeDetector([(0, 0, 5, 5)])
        processor._detector = detector
        image = Image.new("RGB", (20, 16), "red")
        image.paste("blue", (10, 0, 20, 16))
        prepared = processor.prepare(image)
        self.assertEqual(detector.calls, 0)
        self.assertEqual(prepared.getpixel((3, 8)), (255, 0, 0))
        self.assertEqual(prepared.getpixel((16, 8)), (0, 0, 255))

    def test_serialization_clears_detector_cache_for_spawn(self):
        processor = FacePreprocessor(self.config(crop_face=True))
        processor._detector = FakeDetector([(0, 0, 10, 10)])
        restored = pickle.loads(pickle.dumps(processor))
        self.assertIsNone(restored._detector)
        self.assertEqual(restored.config, processor.config)
        self.assertEqual(tuple(restored.tensor(Image.new("RGB", (20, 16)),
                                               already_cropped=True).shape), (3, 16, 20))

    def test_no_face_error_and_keep_are_explicit(self):
        image = Image.new("RGB", (20, 16), "red")
        strict = FacePreprocessor(self.config(crop_face=True, no_face="error"))
        strict._detector = FakeDetector([])
        with self.assertRaises((ValueError, RuntimeError)):
            strict(image)
        keep = FacePreprocessor(self.config(crop_face=True, no_face="keep"))
        keep._detector = FakeDetector([])
        result = keep.prepare(image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (20, 16))
        self.assertEqual(result.getpixel((10, 8)), (255, 0, 0))

    def test_multiple_faces_are_rejected_for_labeled_image(self):
        processor = FacePreprocessor(self.config(crop_face=True), training=True)
        processor._detector = FakeDetector([(0, 0, 10, 10), (10, 0, 10, 10)])
        with self.assertRaises((ValueError, RuntimeError)):
            processor(Image.new("RGB", (20, 16)))

    def test_single_face_crop_preserves_bounded_margin(self):
        detector = FaceDetector()
        image = Image.new("RGB", (100, 80), "red")
        self.assertEqual(detector.crop(image, (50, 40, 40, 30), margin=.2).size, (56, 42))
        self.assertEqual(detector.crop(image, (0, 0, 20, 20), margin=.2).size, (24, 24))

    def test_preprocess_fingerprint_is_stable_and_configuration_sensitive(self):
        config = self.config()
        reversed_config = dict(reversed(list(config.items())))
        self.assertEqual(preprocess_id(config), preprocess_id(reversed_config))
        self.assertNotEqual(preprocess_id(config), preprocess_id(self.config(size=[32, 32])))
        self.assertNotEqual(preprocess_id(config), preprocess_id(self.config(normalization="symmetric")))
        self.assertNotEqual(preprocess_id(config), preprocess_id(self.config(crop_face=True)))

    def test_corrupt_image_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises((ValueError, RuntimeError, OSError)):
                FacePreprocessor(self.config())(path)


if __name__ == "__main__":
    unittest.main()
