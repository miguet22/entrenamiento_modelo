import unittest
from unittest.mock import patch
import config
from detector import CrashDetector


class InferenceThresholdTests(unittest.TestCase):
    def test_inference_does_not_discard_low_confidence_crash_candidates(self):
        with patch.object(CrashDetector, '_init_detector'), patch.dict(
                config.CLASS_CONFIDENCE_THRESHOLDS, crash=.10):
            detector = CrashDetector(conf_threshold=.25)
        self.assertLessEqual(detector.conf_threshold, .10)

    def test_other_classes_keep_their_own_thresholds(self):
        self.assertFalse(config.passes_detection_threshold(
            dict(class_name='car', confidence=.70)))
        with patch.dict(config.CLASS_CONFIDENCE_THRESHOLDS, crash=.10):
            self.assertTrue(config.passes_detection_threshold(
                dict(class_name='crash', confidence=.15)))
