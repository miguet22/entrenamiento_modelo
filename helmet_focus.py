"""Segunda pasada de casco en otro proceso, con una cola acotada."""

import multiprocessing as mp
from queue import Empty, Full
import time

import config
from helmet_monitor import HelmetMonitor, box_iou


def motorcycle_crops(frame, detections):
    """Recortar desde los pixels originales incluyendo cabezas sobre la moto."""
    height, width = frame.shape[:2]
    motos = sorted((d for d in detections
                    if d['class_name'] == 'moto'
                    and config.passes_detection_threshold(d)),
                   key=lambda d: d['confidence'], reverse=True)
    crops = []
    for moto in motos:
        if any(box_iou(moto['bbox'], item[0]['bbox']) > 0.7 for item in crops):
            continue
        x1, y1, x2, y2 = moto['bbox']
        w, h = x2 - x1, y2 - y1
        left = max(0, int(x1 - w * 0.25))
        top = max(0, int(y1 - h * config.HELMET_ABOVE_HEIGHT))
        right = min(width, int(x2 + w * 0.25))
        bottom = min(height, int(y2 + h * 0.1))
        if w > 0 and h > 0 and right > left and bottom > top:
            crops.append((moto, (left, top), frame[top:bottom, left:right].copy()))
        if len(crops) >= config.HELMET_FOCUS_MAX_MOTOS:
            break
    return crops


def focused_detections(detector, crops):
    detections = [item[0] for item in crops]
    heads = []
    for _, (left, top), image in crops:
        for head in detector.predict_frame(image):
            if (head['class_name'] not in ('casco', 'no_casco')
                    or not config.passes_detection_threshold(head)):
                continue
            x1, y1, x2, y2 = head['bbox']
            head = dict(head, bbox=[x1 + left, y1 + top, x2 + left, y2 + top])
            # Un ocupante puede aparecer en dos recortes vecinos: conservar
            # una unica deteccion, incluso si las etiquetas discrepan.
            overlapping = next((i for i, old in enumerate(heads)
                                if box_iou(old['bbox'], head['bbox']) > 0.5), None)
            if overlapping is None:
                heads.append(head)
            elif head['confidence'] > heads[overlapping]['confidence']:
                heads[overlapping] = head
    return detections + heads


def _worker(jobs, results, stop, ready):
    try:
        print('[CASCO] Inicializando modelo paralelo...', flush=True)
        from detector import CrashDetector
        detector = CrashDetector(image_size=config.HELMET_FOCUS_IMAGE_SIZE,
                                 cpu_threads=config.HELMET_FOCUS_CPU_THREADS)
        if detector.model_type != 'yolo':
            raise RuntimeError('El enfoque de casco requiere el modelo YOLO local')
        print('[CASCO] Preparando inferencia...', flush=True)
        import numpy as np
        detector.predict_frame(np.zeros((416, 416, 3), dtype=np.uint8))
        ready.set()
        print('[CASCO] Modelo paralelo listo.', flush=True)
        monitor = HelmetMonitor(
            confidence=config.HELMET_CONFIDENCE_THRESHOLD,
            min_frames=config.HELMET_CONFIRM_FRAMES,
            confirm_seconds=config.HELMET_CONFIRM_SECONDS,
            cooldown_seconds=config.HELMET_COOLDOWN_SECONDS,
            clear_seconds=config.HELMET_CLEAR_SECONDS,
            above_height=config.HELMET_ABOVE_HEIGHT,
            side_margin=config.HELMET_SIDE_MARGIN,
            min_iou=config.MOTO_TRACK_MIN_IOU)
        previous_time = None
        previous_epoch = None
        previous_session = None
        while not stop.is_set():
            try:
                job = jobs.get(timeout=0.1)
            except Empty:
                continue
            now = job['event_time']
            if job['session'] != previous_session:
                monitor.tracks.clear()
                monitor.next_id = 1
                previous_session = job['session']
            if (job['epoch'] != previous_epoch or previous_time is not None
                    and now - previous_time > config.HELMET_FOCUS_MAX_SAMPLE_GAP):
                monitor.reset_candidates()
            previous_time, previous_epoch = now, job['epoch']
            crops = job.pop('crops')
            detections = focused_detections(detector, crops)
            job['detections'] = detections
            job['preview'] = None
            if crops:
                import cv2
                _, (left, top), preview = crops[0]
                for head in detections:
                    if head['class_name'] not in ('casco', 'no_casco'):
                        continue
                    x1, y1, x2, y2 = head['bbox']
                    color = config.CLASS_COLORS[head['class_name']]
                    cv2.rectangle(preview, (x1-left, y1-top),
                                  (x2-left, y2-top), color, 2)
                    cv2.putText(preview, head['class_name'].upper(),
                                (max(0, x1-left), max(15, y1-top-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                job['preview'] = preview
            job['reports'] = monitor.update(detections, now)
            if not job['reports']:
                job['image'] = None
            while not stop.is_set():
                try:
                    results.put(job, timeout=0.1)
                    break
                except Full:
                    pass
    except Exception as exc:
        ready.set()
        try:
            results.put({'error': str(exc)}, timeout=1)
        except Full:
            pass


class HelmetFocus:
    def __init__(self):
        context = mp.get_context('spawn')
        self.jobs = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=4)
        self.stop = context.Event()
        self.ready = context.Event()
        self.epoch = 0
        self.session = 0
        self.serial = 0
        self.last_done = 0
        self.process = context.Process(target=_worker,
                                       args=(self.jobs, self.results, self.stop, self.ready),
                                       daemon=True)
        self.process.start()
        if not self.ready.wait(timeout=15):
            self.close()
            raise RuntimeError('El proceso de casco no pudo inicializarse')

    def reset(self):
        self.epoch += 1

    def new_session(self):
        self.reset()
        self.session += 1

    def submit(self, frame, detections, event_time, time_str, frame_idx):
        crops = motorcycle_crops(frame, detections)
        job = dict(crops=crops, event_time=event_time, time=time_str,
                   frame=frame_idx, image=frame.copy() if crops else None,
                   epoch=self.epoch, session=self.session, serial=self.serial + 1)
        # Reemplazar el trabajo pendiente por el mas reciente sin esperar a IA.
        try:
            self.jobs.get_nowait()
        except Empty:
            pass
        try:
            self.jobs.put_nowait(job)
            self.serial += 1
        except Full:
            pass

    def poll(self):
        items = []
        while True:
            try:
                item = self.results.get_nowait()
            except Empty:
                return items
            if 'error' in item or item['epoch'] == self.epoch:
                items.append(item)
            if 'serial' in item:
                self.last_done = max(self.last_done, item['serial'])

    def finish(self, timeout=3):
        deadline = time.monotonic() + timeout
        items = []
        while time.monotonic() < deadline:
            items.extend(self.poll())
            if self.last_done >= self.serial or not self.process.is_alive():
                break
            time.sleep(0.02)
        items.extend(self.poll())
        return items

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        for queue in (self.jobs, self.results):
            queue.cancel_join_thread()
            queue.close()
