"""Confirmacion temporal de choques, independiente del modelo y del reloj."""


class CrashFilter:
    def __init__(self, min_frames=5, confirm_seconds=0.5,
                 cooldown_seconds=10.0, clear_seconds=2.0):
        self.min_frames = min_frames
        self.confirm_seconds = confirm_seconds
        self.cooldown_seconds = cooldown_seconds
        self.clear_seconds = clear_seconds
        self.active = False
        self.last_alert = float('-inf')
        self.clear_since = None
        self.reset_candidate()

    def reset_candidate(self):
        self.candidate_since = None
        self.frames = 0

    def update(self, has_crash, now):
        """Devuelve True una sola vez al confirmar un evento nuevo."""
        if not has_crash:
            self.reset_candidate()
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= self.clear_seconds:
                self.active = False
            return False

        # Una pausa suficientemente larga termina el evento anterior incluso
        # si la siguiente muestra ya vuelve a contener una deteccion.
        if self.clear_since is not None:
            if now - self.clear_since >= self.clear_seconds:
                self.active = False
            self.clear_since = None

        if self.active or now - self.last_alert < self.cooldown_seconds:
            self.reset_candidate()
            return False
        if self.candidate_since is None:
            self.candidate_since = now
        self.frames += 1
        if (self.frames >= self.min_frames
                and now - self.candidate_since >= self.confirm_seconds):
            self.active = True
            self.last_alert = now
            self.reset_candidate()
            return True
        return False
