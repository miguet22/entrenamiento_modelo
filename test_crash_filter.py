import unittest

from crash_filter import CrashFilter


class CrashFilterTests(unittest.TestCase):
    def test_isolated_and_intermittent_frames_do_not_alert(self):
        f = CrashFilter()
        for i in range(100):
            self.assertFalse(f.update(i % 2 == 0, i * 0.1))

    def test_requires_both_frames_and_duration(self):
        f = CrashFilter()
        for t in (0, 0.01, 0.02, 0.03, 0.04):
            self.assertFalse(f.update(True, t))
        self.assertTrue(f.update(True, 0.5))
        f = CrashFilter()
        self.assertFalse(f.update(True, 0))
        self.assertFalse(f.update(True, 1))

    def test_persistent_crash_alerts_only_once(self):
        f = CrashFilter()
        alerts = [f.update(True, i * 0.25) for i in range(200)]
        self.assertEqual(sum(alerts), 1)

    def test_clear_and_cooldown_required_before_fresh_confirmation(self):
        f = CrashFilter()
        for i in range(5):
            alerted = f.update(True, i * 0.25)
        self.assertTrue(alerted)
        f.update(False, 2)
        f.update(False, 4)
        self.assertFalse(f.active)
        for t in (5, 6, 7, 8, 9, 10):
            self.assertFalse(f.update(True, t))
        for t in (11, 11.25, 11.5, 11.75):
            self.assertFalse(f.update(True, t))
        self.assertTrue(f.update(True, 12))

    def test_brief_gap_does_not_rearm_event(self):
        f = CrashFilter()
        for i in range(5):
            f.update(True, i * 0.25)
        f.update(False, 15)
        self.assertFalse(f.update(True, 16))
        self.assertTrue(f.active)


if __name__ == '__main__':
    unittest.main()
