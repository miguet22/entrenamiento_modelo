import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np
import config

from helmet_focus import motorcycle_crops, focused_detections
from helmet_monitor import HelmetMonitor


def detection(name, bbox, confidence=0.9):
    return dict(class_name=name, bbox=bbox, confidence=confidence, is_crash=False)


class HelmetFocusTests(unittest.TestCase):
    @patch.dict(config.CLASS_CONFIDENCE_THRESHOLDS,
                dict(car=0.75, moto=0.70, casco=0.50, no_casco=0.50, crash=0.60))
    def test_class_thresholds_require_strictly_higher_confidence(self):
        for name, threshold in [('car', 0.75), ('moto', 0.70),
                                ('casco', 0.50), ('no_casco', 0.50),
                                ('crash', 0.60)]:
            with self.subTest(name=name):
                for confidence in (threshold - 0.01, threshold):
                    self.assertFalse(config.passes_detection_threshold(
                        detection(name, [0, 0, 10, 10], confidence)))
                self.assertTrue(config.passes_detection_threshold(
                    detection(name, [0, 0, 10, 10], threshold + 0.01)))

    @patch.dict(config.CLASS_CONFIDENCE_THRESHOLDS, dict(casco=0.50, no_casco=0.50))
    def test_focus_accepts_head_above_fifty_and_rejects_equal_threshold(self):
        detector = Mock()
        detector.predict_frame.return_value = [
            detection('casco', [0, 0, 10, 10], 0.50),
            detection('no_casco', [20, 0, 30, 10], 0.51)]
        moto = detection('moto', [0, 20, 100, 100], 0.71)
        results = focused_detections(detector, [(moto, (0, 0), None)])
        self.assertEqual([d['class_name'] for d in results], ['moto', 'no_casco'])
        monitor = HelmetMonitor(confidence=config.HELMET_CONFIDENCE_THRESHOLD,
                                min_frames=2, confirm_seconds=0.08)
        self.assertEqual(monitor.update(results, 0), [])
        self.assertEqual(len(monitor.update(results, 0.1)), 1)

    def test_crop_includes_head_and_preserves_original_pixels(self):
        frame = np.zeros((400, 500, 3), dtype=np.uint8)
        frame[100, 150] = [10, 20, 30]
        moto = detection('moto', [100, 200, 200, 300])
        crops = motorcycle_crops(frame, [moto])
        _, (left, top), image = crops[0]
        self.assertLessEqual(top, 100)
        self.assertEqual(image[100-top, 150-left].tolist(), [10, 20, 30])
        frame[:] = 255
        self.assertEqual(image[100-top, 150-left].tolist(), [10, 20, 30])

    def test_crop_clips_edges_and_ignores_other_vehicles(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crops = motorcycle_crops(frame, [detection('car', [0, 0, 20, 20]),
                                        detection('moto', [0, 0, 100, 100])])
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0][1], (0, 0))
        self.assertEqual(crops[0][2].shape, frame.shape)

    def test_heads_return_to_source_coordinates_and_duplicates_merge(self):
        detector = Mock()
        detector.predict_frame.side_effect = [
            [detection('no_casco', [10, 10, 30, 30], 0.8),
             detection('crash', [0, 0, 100, 100])],
            [detection('no_casco', [5, 10, 25, 30], 0.9)]]
        crops = [(detection('moto', [100, 100, 150, 200]), (100, 50), None),
                 (detection('moto', [150, 100, 200, 200]), (105, 50), None)]
        detections = focused_detections(detector, crops)
        heads = [d for d in detections if d['class_name'] == 'no_casco']
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0]['bbox'], [110, 60, 130, 80])
        self.assertEqual(heads[0]['confidence'], 0.9)
        self.assertFalse(any(d['is_crash'] for d in detections))

    def test_fast_pass_requires_two_real_positive_samples(self):
        monitor = HelmetMonitor(min_frames=2, confirm_seconds=0.08)
        moto = detection('moto', [100, 200, 200, 300])
        head = detection('no_casco', [120, 100, 140, 120])
        self.assertEqual(monitor.update([moto, head], 0), [])
        self.assertEqual(len(monitor.update([moto, head], 0.1)), 1)
        self.assertEqual(monitor.update([moto, head], 0.2), [])
        other = HelmetMonitor(min_frames=2, confirm_seconds=0.08)
        self.assertEqual(other.update([moto], 0), [])
        self.assertEqual(other.update([moto], 0.1), [])

    def test_parallel_report_uses_original_frame_and_timestamp(self):
        import main
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {
            main.cv2.CAP_PROP_FRAME_COUNT: 1,
            main.cv2.CAP_PROP_FPS: 30,
            main.cv2.CAP_PROP_FRAME_WIDTH: 640,
            main.cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }[prop]
        capture.read.side_effect = [(True, object()), (False, None)]
        detector = Mock(model_type='yolo')
        detector.predict_frame.return_value = []
        source_image = object()
        focus = Mock()
        focus.poll.return_value = []
        # Un resultado llega al terminar el video, correspondiente a otro cuadro.
        focus.finish.return_value = [dict(reports=[dict(moto_id=7,
            violation_confidence=0.91)], image=source_image, time='00:00.50',
            frame=15, preview=None)]
        output = StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, 'get_source_selection', return_value='1'))
            stack.enter_context(patch.object(main, 'select_video_file', return_value='test.mp4'))
            stack.enter_context(patch.object(main, 'CrashDetector', return_value=detector))
            stack.enter_context(patch.object(main, 'HelmetFocus', return_value=focus))
            stack.enter_context(patch.object(main.cv2, 'VideoCapture', return_value=capture))
            snapshot = stack.enter_context(patch.object(main.cv2, 'imwrite', return_value=True))
            stack.enter_context(patch.object(main.os, 'makedirs'))
            stack.enter_context(patch.multiple(main.config, SHOW_PREVIEW=False,
                REALTIME_VIDEO_PLAYBACK=False, SAVE_OUTPUT_VIDEO=False,
                AUTO_SAVE_CRASH_SNAPSHOT=False, AUTO_SAVE_HELMET_SNAPSHOT=True))
            stack.enter_context(redirect_stdout(output))
            main.main()
        snapshot.assert_called_once()
        self.assertIs(snapshot.call_args.args[1], source_image)
        self.assertIn('_moto7_f15.jpg', snapshot.call_args.args[0])
        self.assertIn('Hora/Tiempo: 00:00.50 | Frame #15', output.getvalue())
        focus.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
