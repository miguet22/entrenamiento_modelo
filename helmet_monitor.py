"""Asocia cascos a motos y confirma infracciones por moto entre frames."""

from crash_filter import CrashFilter


def associate_riders(detections, confidence=0.6, above_height=1.5,
                     side_margin=0.15):
    """Cada cabeza se asigna a una sola moto cercana, en su zona superior.

    No observar un casco no prueba una infraccion: se exige no_casco.
    """
    motos = [dict(d, riders=[]) for d in detections
             if d.get('class_name', '').lower() == 'moto'
             and d.get('confidence', 0) >= confidence and d.get('bbox')]
    for head in detections:
        if (head.get('class_name', '').lower() not in ('casco', 'no_casco')
                or head.get('confidence', 0) < confidence
                or not head.get('bbox')):
            continue
        hx1, hy1, hx2, hy2 = head['bbox']
        hx, hy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        candidates = []
        for index, moto in enumerate(motos):
            x1, y1, x2, y2 = moto['bbox']
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            if (x1 - side_margin * w <= hx <= x2 + side_margin * w
                    and y1 - above_height * h <= hy <= y1 + 0.5 * h):
                distance = ((hx - (x1 + x2) / 2) / w) ** 2
                distance += ((hy - y1) / (h * above_height)) ** 2
                candidates.append((distance, index))
        if candidates:
            _, index = min(candidates)
            motos[index]['riders'].append(head)

    for moto in motos:
        unsafe = [d for d in moto['riders']
                  if d['class_name'].lower() == 'no_casco']
        moto['violation'] = bool(unsafe)
        moto['violation_confidence'] = (
            min(moto['confidence'], max(d['confidence'] for d in unsafe))
            if unsafe else 0.0)
    return motos


def box_iou(a, b):
    intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0, min(a[3], b[3]) - max(a[1], b[1])))
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


class HelmetMonitor:
    def __init__(self, confidence=0.6, min_frames=5, confirm_seconds=0.5,
                 cooldown_seconds=10.0, clear_seconds=2.0,
                 above_height=1.5, side_margin=0.15, min_iou=0.2):
        self.confidence = confidence
        self.above_height = above_height
        self.side_margin = side_margin
        self.min_iou = min_iou
        self.retention = max(cooldown_seconds, clear_seconds)
        self.filter_options = dict(min_frames=min_frames,
                                   confirm_seconds=confirm_seconds,
                                   cooldown_seconds=cooldown_seconds,
                                   clear_seconds=clear_seconds)
        self.tracks = {}
        self.next_id = 1

    def reset_candidates(self):
        """Una interrupcion de video invalida las confirmaciones pendientes."""
        for track in self.tracks.values():
            track['filter'].reset_candidate()

    def update(self, detections, now):
        """Devuelve solo infracciones nuevas, con un filtro por moto.

        Seguimiento geometrico por solapamiento; no identifica vehiculos
        que salen de escena y vuelven, o que se ocultan completamente.
        """
        motos = associate_riders(detections, self.confidence,
                                 self.above_height, self.side_margin)
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track['last_seen'] <= self.retention}
        # Emparejamiento uno a uno: primero las cajas de mayor solapamiento.
        matches = []
        for key, track in self.tracks.items():
            for index, moto in enumerate(motos):
                score = box_iou(track['bbox'], moto['bbox'])
                if score >= self.min_iou:
                    matches.append((score, key, index))
        assignments = {}
        used_tracks = set()
        for _, key, index in sorted(matches, reverse=True):
            if key not in used_tracks and index not in assignments:
                assignments[index] = key
                used_tracks.add(key)

        for key, track in self.tracks.items():
            if key not in used_tracks:
                track['filter'].update(False, now)

        reports = []
        for index, moto in enumerate(motos):
            key = assignments.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = {'filter': CrashFilter(**self.filter_options)}
            track = self.tracks[key]
            track['bbox'] = moto['bbox']
            track['last_seen'] = now
            if track['filter'].update(moto['violation'], now):
                reports.append(dict(moto, moto_id=key))
        return reports
