import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from helmet_monitor import HelmetMonitor, associate_riders


def detection(name, bbox, confidence=0.9):
    return dict(class_name=name, bbox=bbox, confidence=confidence, is_crash=False)


MOTO = detection('moto', [100, 200, 200, 300])
HELMET = detection('casco', [115, 120, 135, 145])
NO_HELMET = detection('no_casco', [160, 125, 180, 150])


class HelmetAssociationTests(unittest.TestCase):
    def test_one_of_two_riders_without_helmet_is_violation_in_any_order(self):
        for heads in ([HELMET, NO_HELMET], [NO_HELMET, HELMET]):
            moto = associate_riders([MOTO, *heads])[0]
            self.assertTrue(moto['violation'])
            self.assertEqual(len(moto['riders']), 2)

    def test_both_helmeted_or_no_head_detection_are_not_violations(self):
        for heads in ([], [HELMET], [HELMET, detection('casco', NO_HELMET['bbox'])]):
            self.assertFalse(associate_riders([MOTO, *heads])[0]['violation'])

    def test_head_requires_moto_and_position_above_it(self):
        self.assertEqual(associate_riders([NO_HELMET]), [])
        for bbox in ([300, 125, 320, 150], [150, 310, 170, 330],
                     [150, 0, 170, 20]):
            result = associate_riders([MOTO, detection('no_casco', bbox)])
            self.assertFalse(result[0]['violation'])

    def test_low_confidence_head_or_moto_does_not_trigger(self):
        result = associate_riders([MOTO, dict(NO_HELMET, confidence=0.59)])
        self.assertFalse(result[0]['violation'])
        self.assertEqual(associate_riders([dict(MOTO, confidence=0.59), NO_HELMET]), [])

    def test_neighboring_motos_do_not_share_head(self):
        second = detection('moto', [190, 200, 290, 300])
        head = detection('no_casco', [188, 125, 202, 150])
        result = associate_riders([MOTO, second, head])
        self.assertEqual(sum(m['violation'] for m in result), 1)
        self.assertEqual(sum(len(m['riders']) for m in result), 1)


class HelmetMonitorTests(unittest.TestCase):
    def test_single_and_intermittent_frames_are_suppressed(self):
        monitor = HelmetMonitor()
        for i in range(20):
            heads = [NO_HELMET] if i % 2 == 0 else [HELMET]
            self.assertEqual(monitor.update([MOTO, *heads], i * 0.25), [])

    def test_two_riders_confirm_once_and_persistent_violation_does_not_repeat(self):
        monitor = HelmetMonitor()
        reports = []
        for i in range(100):
            reports.extend(monitor.update([MOTO, HELMET, NO_HELMET], i * 0.25))
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]['moto_id'], 1)

    def test_two_motos_have_independent_confirmation_and_cooldowns(self):
        monitor = HelmetMonitor()
        second = detection('moto', [300, 200, 400, 300])
        second_head = detection('no_casco', [330, 125, 350, 150])
        reports = []
        for i in range(9):
            heads = [NO_HELMET] + ([second_head] if i >= 4 else [])
            # Cambiar el orden no debe cambiar la identidad de las motos.
            motos = [MOTO, second] if i % 2 else [second, MOTO]
            reports.extend(monitor.update([*motos, *heads], i * 0.25))
        self.assertEqual(len(reports), 2)
        self.assertEqual(len({r['moto_id'] for r in reports}), 2)

    def test_evidence_on_different_motos_cannot_accumulate(self):
        monitor = HelmetMonitor()
        for i in range(20):
            shift = (i % 2) * 300
            items = [dict(d, bbox=[d['bbox'][0] + shift, d['bbox'][1],
                                   d['bbox'][2] + shift, d['bbox'][3]])
                     for d in (MOTO, NO_HELMET)]
            self.assertEqual(monitor.update(items, i * 0.25), [])

    def test_motion_preserves_track(self):
        monitor = HelmetMonitor()
        reports = []
        for i in range(20):
            items = [dict(d, bbox=[d['bbox'][0] + i * 5, d['bbox'][1],
                                   d['bbox'][2] + i * 5, d['bbox'][3]])
                     for d in (MOTO, NO_HELMET)]
            reports.extend(monitor.update(items, i * 0.25))
        self.assertEqual(len(reports), 1)
        self.assertEqual(len(monitor.tracks), 1)

    def test_cooldown_and_fresh_confirmation_after_clear(self):
        monitor = HelmetMonitor()
        reports = []
        for i in range(53):
            now = i * 0.25
            heads = [HELMET] if 2 <= now <= 4 else [NO_HELMET]
            for report in monitor.update([MOTO, *heads], now):
                reports.append((now, report['moto_id']))
        self.assertEqual(reports, [(1.0, 1), (12.0, 1)])

    def test_missing_moto_and_stream_interruption_reset_candidate(self):
        for interrupt in ('missing', 'stream'):
            monitor = HelmetMonitor()
            for i in range(4):
                self.assertEqual(monitor.update([MOTO, NO_HELMET], i * 0.25), [])
            if interrupt == 'missing':
                monitor.update([], 1)
            else:
                monitor.reset_candidates()
            self.assertEqual(monitor.update([MOTO, NO_HELMET], 1.25), [])

    def test_old_tracks_are_expired(self):
        monitor = HelmetMonitor()
        monitor.update([MOTO], 0)
        monitor.update([], 11)
        self.assertEqual(monitor.tracks, {})


class HelmetReportingTests(unittest.TestCase):
    def test_main_reports_once_and_saves_one_snapshot(self):
        import main

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {
            main.cv2.CAP_PROP_FRAME_COUNT: 24,
            main.cv2.CAP_PROP_FPS: 8,
            main.cv2.CAP_PROP_FRAME_WIDTH: 640,
            main.cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }[prop]
        capture.read.side_effect = [(True, object())] * 24 + [(False, None)]
        detector = Mock()
        detector.predict_frame.return_value = [MOTO, HELMET, NO_HELMET]
        output = StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, 'get_source_selection', return_value='1'))
            stack.enter_context(patch.object(main, 'select_video_file', return_value='test.mp4'))
            stack.enter_context(patch.object(main, 'CrashDetector', return_value=detector))
            stack.enter_context(patch.object(main.cv2, 'VideoCapture', return_value=capture))
            snapshot = stack.enter_context(patch.object(main.cv2, 'imwrite'))
            stack.enter_context(patch.object(main.os, 'makedirs'))
            stack.enter_context(patch.multiple(main.config, SHOW_PREVIEW=False,
                                               SAVE_OUTPUT_VIDEO=False,
                                               AUTO_SAVE_CRASH_SNAPSHOT=True,
                                               AUTO_SAVE_HELMET_SNAPSHOT=True))
            stack.enter_context(redirect_stdout(output))
            main.main()
        self.assertEqual(output.getvalue().count('[INFRACCION: SIN CASCO]'), 1)
        self.assertIn('Total de infracciones por falta de casco: 1', output.getvalue())
        self.assertIn('Total de eventos de choque detectados: 0', output.getvalue())
        snapshot.assert_called_once()
        self.assertTrue(snapshot.call_args.args[0].startswith(main.config.HELMET_SNAPSHOTS_DIR))
        self.assertIn('_moto1_', snapshot.call_args.args[0])
        capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
