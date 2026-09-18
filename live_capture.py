"""Lectura continua de camara, conservando solo el cuadro mas reciente."""

from threading import Condition, Event, Thread


class LatestFrameCapture:
    def __init__(self, capture):
        self.capture = capture
        self.condition = Condition()
        self.stopped = Event()
        self.latest = None
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopped.is_set():
                result = self.capture.read()
                with self.condition:
                    self.latest = result
                    self.condition.notify_all()
                if not result[0]:
                    self.stopped.wait(0.1)
        finally:
            self.capture.release()

    def read(self):
        with self.condition:
            self.condition.wait_for(
                lambda: self.latest is not None or self.stopped.is_set(),
                timeout=2.0,
            )
            result = self.latest
            self.latest = None
            return result if result is not None else (False, None)

    def release(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        # El hilo libera la camara: evita release() durante un read() nativo.
        self.thread.join(timeout=2.0)
