"""API local de deteccion en vivo. Un modelo y una transmision activa."""

import asyncio
import base64
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import logging
import math
import os
from pathlib import Path
import re
import struct
import time
from uuid import uuid4

import cv2
import anyio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import numpy as np

import config
from crash_filter import CrashFilter
from detector import CrashDetector
from helmet_focus import HelmetFocus
from helmet_monitor import HelmetMonitor

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
MAX_PACKET_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_294_400


@dataclass
class FramePacket:
    frame_id: int
    timestamp_ms: float
    encoded: bytes


def parse_packet(message):
    """Binario: longitud JSON uint32 BE + metadata JSON + imagen JPEG/PNG."""
    if message.get('bytes') is not None:
        data = message['bytes']
        if len(data) > MAX_PACKET_BYTES or len(data) < 5:
            raise ValueError('Tamaño de paquete invalido')
        size = struct.unpack('!I', data[:4])[0]
        if not 1 <= size <= 2048 or 4 + size >= len(data):
            raise ValueError('Cabecera binaria invalida')
        metadata = json.loads(data[4:4 + size])
        encoded = data[4 + size:]
    else:
        text = message.get('text', '')
        if len(text) > MAX_PACKET_BYTES:
            raise ValueError('Paquete demasiado grande')
        metadata = json.loads(text)
        if not isinstance(metadata, dict):
            raise ValueError('Se requiere un objeto JSON')
        encoded = base64.b64decode(metadata.get('image_base64', ''), validate=True)
    if not isinstance(metadata, dict) or metadata.get('type', 'frame') != 'frame':
        raise ValueError('Se requiere un mensaje frame')
    frame_id = metadata.get('frame_id')
    timestamp = metadata.get('timestamp_ms')
    if type(frame_id) is not int or not 0 <= frame_id <= 2**53 - 1:
        raise ValueError('frame_id debe ser un entero no negativo')
    if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
            or timestamp < 0):
        raise ValueError('timestamp_ms debe ser un numero finito no negativo')
    if not encoded:
        raise ValueError('Imagen vacia')
    return FramePacket(frame_id, float(timestamp), encoded)


def decode_image(encoded):
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('La imagen no es JPEG/PNG valido')
    if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError('La imagen supera el limite de 3840x2160 pixels')
    return frame


def make_helmet_monitor():
    return HelmetMonitor(
        confidence=config.HELMET_CONFIDENCE_THRESHOLD,
        min_frames=config.HELMET_CONFIRM_FRAMES,
        confirm_seconds=config.HELMET_CONFIRM_SECONDS,
        cooldown_seconds=config.HELMET_COOLDOWN_SECONDS,
        clear_seconds=config.HELMET_CLEAR_SECONDS,
        above_height=config.HELMET_ABOVE_HEIGHT,
        side_margin=config.HELMET_SIDE_MARGIN,
        min_iou=config.MOTO_TRACK_MIN_IOU)


class DetectionRuntime:
    def __init__(self, detector_factory=CrashDetector, focus_factory=HelmetFocus,
                 data_dir=None):
        self.detector_factory = detector_factory
        self.focus_factory = focus_factory
        self.data_dir = Path(data_dir or ROOT / 'service_data')
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='detector')
        self.detector = None
        self.focus = None
        self.active_stream = None
        self.events = deque(maxlen=200)
        self.started = time.monotonic()
        self.ready = False
        self.fatal_error = None
        self.on_fatal = None

    def fail(self, error):
        self.fatal_error = str(error)
        self.ready = False
        if self.on_fatal is not None:
            self.on_fatal()

    async def run(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, function, *args)

    def initialize(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'snapshots').mkdir(exist_ok=True)
        self.detector = self.detector_factory()
        if self.detector.model_type != 'yolo':
            raise RuntimeError('No se pudo cargar el modelo YOLO local')
        self.detector.predict_frame(np.zeros((416, 416, 3), dtype=np.uint8))
        if config.HELMET_FOCUS_ENABLED:
            self.focus = self.focus_factory()
            for result in self.focus.poll():
                if 'error' in result:
                    raise RuntimeError(result['error'])
        self.ready = True

    def shutdown(self):
        self.ready = False
        if self.focus is not None:
            self.focus.close()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def save_snapshot(self, image, kind):
        if image is None:
            return None
        filename = f'{kind}_{uuid4().hex}.jpg'
        try:
            if not cv2.imwrite(str(self.data_dir / 'snapshots' / filename), image):
                raise OSError('No se pudo guardar la captura')
        except (OSError, cv2.error):
            LOGGER.exception('Error guardando captura')
            return None
        return f'/snapshots/{filename}'


class LiveSession:
    def __init__(self, runtime, websocket):
        self.runtime = runtime
        self.websocket = websocket
        self.stream_id = uuid4().hex
        self.pending = None
        self.available = asyncio.Event()
        self.outgoing = asyncio.Queue(maxsize=16)
        self.metadata = OrderedDict()
        self.last_frame_id = -1
        self.last_timestamp = -1
        self.last_processed_time = None
        self.dropped = 0
        self.sequence = 0
        self.helmet_monitor = make_helmet_monitor()
        self.crash_filter = CrashFilter(
            min_frames=config.CRASH_CONFIRM_FRAMES,
            confirm_seconds=config.CRASH_CONFIRM_SECONDS,
            cooldown_seconds=config.CRASH_COOLDOWN_SECONDS,
            clear_seconds=config.CRASH_CLEAR_SECONDS)

    def envelope(self, kind, packet, **fields):
        return dict(type=kind, stream_id=self.stream_id,
                    frame_id=packet.frame_id, timestamp_ms=packet.timestamp_ms,
                    **fields)

    async def receive(self):
        while True:
            message = await self.websocket.receive()
            if message['type'] == 'websocket.disconnect':
                return
            try:
                packet = parse_packet(message)
                if (packet.frame_id <= self.last_frame_id
                        or packet.timestamp_ms <= self.last_timestamp):
                    raise ValueError('frame_id y timestamp_ms deben aumentar en cada cuadro')
            except (ValueError, TypeError, UnicodeError, json.JSONDecodeError,
                    OverflowError) as exc:
                await self.outgoing.put(dict(type='error', code='invalid_frame',
                                             message=str(exc)))
                continue
            self.last_frame_id, self.last_timestamp = packet.frame_id, packet.timestamp_ms
            if self.pending is not None:
                self.dropped += 1
            self.pending = packet
            self.available.set()

    async def send(self):
        while True:
            await self.websocket.send_json(await self.outgoing.get())

    async def emit_event(self, packet, kind, image, **fields):
        should_save = (config.AUTO_SAVE_CRASH_SNAPSHOT if kind == 'crash'
                       else config.AUTO_SAVE_HELMET_SNAPSHOT)
        snapshot_url = (await self.runtime.run(self.runtime.save_snapshot, image, kind)
                        if should_save else None)
        event = self.envelope('event', packet, event_id=uuid4().hex,
                              event=kind, snapshot_url=snapshot_url, **fields)
        self.runtime.events.append(event)
        await self.outgoing.put(event)

    async def emit_helmet_reports(self, packet, reports, image):
        for report in reports:
            await self.emit_event(packet, 'no_casco', image,
                                  moto_id=report['moto_id'],
                                  confidence=report['violation_confidence'],
                                  bbox=report['bbox'])

    async def process(self):
        while True:
            await self.available.wait()
            packet, self.pending = self.pending, None
            self.available.clear()
            start = time.perf_counter()
            try:
                image = await self.runtime.run(decode_image, packet.encoded)
            except (ValueError, cv2.error) as exc:
                await self.outgoing.put(self.envelope('error', packet,
                    code='invalid_image', message=str(exc)))
                continue
            detections = await self.runtime.run(self.runtime.detector.predict_frame, image)
            detections = [d for d in detections if config.passes_detection_threshold(d)]
            self.sequence += 1
            self.metadata[self.sequence] = FramePacket(packet.frame_id, packet.timestamp_ms, b'')
            while len(self.metadata) > 256:
                self.metadata.popitem(last=False)
            now = packet.timestamp_ms / 1000
            if (self.last_processed_time is not None
                    and now - self.last_processed_time > config.HELMET_FOCUS_MAX_SAMPLE_GAP):
                self.crash_filter.reset_candidate()
                self.helmet_monitor.reset_candidates()
            self.last_processed_time = now
            if self.runtime.focus is not None:
                self.runtime.focus.submit(image, detections, now,
                                          str(packet.timestamp_ms), self.sequence)
            else:
                await self.emit_helmet_reports(packet,
                    self.helmet_monitor.update(detections, now), image)
            crashes = [d for d in detections if d.get('is_crash')
                       and d['confidence'] >= config.CRASH_CONFIDENCE_THRESHOLD]
            new_crash = self.crash_filter.update(bool(crashes), now)
            await self.outgoing.put(self.envelope('detections', packet,
                width=image.shape[1], height=image.shape[0], detections=detections,
                crash_confirmed=bool(crashes) and self.crash_filter.active,
                processing_ms=round((time.perf_counter() - start) * 1000, 1),
                dropped_frames=self.dropped))
            if new_crash:
                crash = max(crashes, key=lambda item: item['confidence'])
                await self.emit_event(packet, 'crash', image,
                                      confidence=crash['confidence'], bbox=crash['bbox'])

    async def poll_helmets(self):
        while True:
            focus = self.runtime.focus
            if focus is not None:
                for result in focus.poll():
                    if 'error' in result:
                        raise RuntimeError(f"Fallo del proceso de cascos: {result['error']}")
                    packet = self.metadata.get(result['frame'])
                    if packet is None:
                        continue
                    detections = result.get('detections', [])
                    if detections:
                        await self.outgoing.put(self.envelope('helmet_detections', packet,
                            detections=detections))
                    await self.emit_helmet_reports(packet, result['reports'], result['image'])
                if not focus.process.is_alive():
                    raise RuntimeError('El proceso de cascos dejo de responder')
            await asyncio.sleep(0.02)

    async def run(self):
        if self.runtime.focus is not None:
            self.runtime.focus.new_session()
        await self.websocket.send_json(dict(type='ready', stream_id=self.stream_id,
            protocol_version=1, recommended_fps=10, max_packet_bytes=MAX_PACKET_BYTES,
            thresholds=config.CLASS_CONFIDENCE_THRESHOLDS,
            helmet_focus=self.runtime.focus is not None))
        tasks = [asyncio.create_task(function()) for function in
                 (self.receive, self.send, self.process, self.poll_helmets)]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            with anyio.CancelScope(shield=True):
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                # Esperar la inferencia en curso antes de admitir otra sesion.
                await self.runtime.run(lambda: None)
                if self.runtime.focus is not None:
                    self.runtime.focus.new_session()


def create_app(runtime=None, api_key=None):
    runtime = runtime or DetectionRuntime()
    api_key = api_key if api_key is not None else os.environ.get('DETECTION_API_KEY', '')

    @asynccontextmanager
    async def lifespan(app):
        try:
            await runtime.run(runtime.initialize)
            yield
        finally:
            # No usar el mismo executor para shutdown(): esperaria su propio hilo.
            await asyncio.to_thread(runtime.shutdown)

    app = FastAPI(title='Deteccion de video en vivo', version='1.0.0', lifespan=lifespan)
    app.state.runtime = runtime

    def authorized(token):
        return not api_key or hmac.compare_digest(
            (token or '').encode('utf-8'), api_key.encode('utf-8'))

    def authorize_http(request):
        if not authorized(request.headers.get('x-api-key')):
            raise HTTPException(401, 'API key invalida')

    @app.get('/health')
    async def health():
        return dict(status='ready' if runtime.ready else 'starting',
                    active_stream=runtime.active_stream,
                    helmet_focus=runtime.focus is not None,
                    thresholds=config.CLASS_CONFIDENCE_THRESHOLDS,
                    uptime_seconds=round(time.monotonic() - runtime.started, 1))

    @app.get('/events')
    async def events(request: Request):
        authorize_http(request)
        return dict(events=list(runtime.events))

    @app.get('/snapshots/{filename}')
    async def snapshot(filename: str, request: Request):
        authorize_http(request)
        if not re.fullmatch(r'(crash|no_casco)_[0-9a-f]{32}\.jpg', filename):
            raise HTTPException(404, 'Captura no encontrada')
        path = runtime.data_dir / 'snapshots' / filename
        if not path.is_file():
            raise HTTPException(404, 'Captura no encontrada')
        return FileResponse(path, media_type='image/jpeg')

    @app.websocket('/ws/live')
    async def live(websocket: WebSocket):
        if not authorized(websocket.headers.get('x-api-key')
                          or websocket.query_params.get('token')):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if not runtime.ready:
            await websocket.send_json(dict(type='error', code='not_ready',
                message='El detector no esta disponible'))
            await websocket.close(code=1013)
            return
        if runtime.active_stream is not None:
            await websocket.send_json(dict(type='error', code='busy',
                message='Ya hay una transmision activa'))
            await websocket.close(code=1013)
            return
        session = LiveSession(runtime, websocket)
        runtime.active_stream = session.stream_id
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            LOGGER.exception('Fallo procesando la transmision %s', session.stream_id)
            runtime.fail(exc)
            try:
                await websocket.close(code=1011, reason='Error de procesamiento')
            except RuntimeError:
                pass
        finally:
            runtime.active_stream = None

    return app
