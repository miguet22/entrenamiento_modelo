"""Comprueba que la inferencia lenta no acumula cuadros de camara."""

import unittest
from threading import Event

from live_capture import LatestFrameCapture


class FakeCamera:
    def __init__(self):
        self.index = 0
        self.finished = Event()
        self.resume = Event()
        self.released = Event()

    def read(self):
        if self.index < 100:
            self.index += 1
            return True, self.index
        self.finished.set()
        self.resume.wait(5)
        return False, None

    def release(self):
        self.released.set()


class LatestFrameTests(unittest.TestCase):
    def test_delivers_latest_frame_once_and_releases_camera(self):
        camera = FakeCamera()
        capture = LatestFrameCapture(camera)
        try:
            self.assertTrue(camera.finished.wait(2))
            self.assertEqual(capture.read(), (True, 100))
            self.assertIsNone(capture.latest)
        finally:
            capture.stopped.set()
            camera.resume.set()
            capture.release()
        self.assertTrue(camera.released.is_set())
        self.assertFalse(capture.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
