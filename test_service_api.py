import asyncio
import base64
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock, patch

import cv2
from fastapi.testclient import TestClient
import numpy as np

import config
from service_api import DetectionRuntime, FramePacket, LiveSession, create_app, parse_packet


def packet(frame_id, timestamp_ms, encoded=None):
    if encoded is None:
        _, image = cv2.imencode('.jpg', np.zeros((80, 120, 3), dtype=np.uint8))
        encoded = image.tobytes()
    metadata = json.dumps(dict(frame_id=frame_id, timestamp_ms=timestamp_ms)).encode()
    return struct.pack('!I', len(metadata)) + metadata + encoded


class FakeDetector:
    model_type = 'yolo'

    def __init__(self, detections=None):
        self.calls = 0
        self.detections = detections or []
        self.entered = Event()
        self.resume = Event()
        self.block = False

    def predict_frame(self, image):
        self.calls += 1
        if self.block and self.calls == 2:
            self.entered.set()
            if not self.resume.wait(5):
                raise RuntimeError('Timeout en prueba de backpressure')
        return self.detections


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.folder = self.stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).parent))
        self.stack.enter_context(patch.multiple(config, HELMET_FOCUS_ENABLED=False,
            AUTO_SAVE_CRASH_SNAPSHOT=False, AUTO_SAVE_HELMET_SNAPSHOT=False))
        self.addCleanup(self.stack.close)

    def client(self, detector=None, key=''):
        self.detector = detector or FakeDetector()
        self.runtime = DetectionRuntime(detector_factory=lambda: self.detector,
                                        data_dir=self.folder)
        client = TestClient(create_app(self.runtime, api_key=key))
        return self.stack.enter_context(client)

    def test_binary_and_json_packets(self):
        result = parse_packet(dict(bytes=packet(42, 100.5)))
        self.assertEqual((result.frame_id, result.timestamp_ms), (42, 100.5))
        text = json.dumps(dict(frame_id=7, timestamp_ms=200,
                               image_base64=base64.b64encode(result.encoded).decode()))
        self.assertEqual(parse_packet(dict(text=text)).encoded, result.encoded)
        for metadata in [dict(frame_id=True, timestamp_ms=1),
                         dict(frame_id=1, timestamp_ms=float('nan')),
                         dict(frame_id=1, timestamp_ms=-1)]:
            with self.assertRaises(ValueError):
                parse_packet(dict(text=json.dumps(metadata)))

    def test_health_and_roundtrip_with_model_loaded_once(self):
        client = self.client()
        self.assertEqual(client.get('/health').json()['status'], 'ready')
        for _ in range(2):
            with client.websocket_connect('/ws/live') as socket:
                self.assertEqual(socket.receive_json()['type'], 'ready')
                socket.send_bytes(packet(123, 100))
                result = socket.receive_json()
                self.assertEqual(result['type'], 'detections')
                self.assertEqual(result['frame_id'], 123)
                self.assertEqual(result['timestamp_ms'], 100)
                self.assertEqual((result['width'], result['height']), (120, 80))
        self.assertEqual(self.detector.calls, 3)  # calentamiento + dos cuadros

    def test_rejects_second_stream_and_accepts_reconnect(self):
        client = self.client()
        with client.websocket_connect('/ws/live') as first:
            first.receive_json()
            with client.websocket_connect('/ws/live') as second:
                self.assertEqual(second.receive_json()['code'], 'busy')
        with client.websocket_connect('/ws/live') as socket:
            self.assertEqual(socket.receive_json()['type'], 'ready')

    def test_invalid_images_and_order_do_not_close_stream(self):
        client = self.client()
        with client.websocket_connect('/ws/live') as socket:
            socket.receive_json()
            socket.send_bytes(packet(1, 100, b'no es JPEG'))
            self.assertEqual(socket.receive_json()['code'], 'invalid_image')
            socket.send_bytes(packet(1, 100))
            self.assertEqual(socket.receive_json()['code'], 'invalid_frame')
            socket.send_bytes(packet(2, 200))
            self.assertEqual(socket.receive_json()['frame_id'], 2)

    def test_busy_inference_keeps_only_latest_pending_frame(self):
        detector = FakeDetector()
        detector.block = True
        client = self.client(detector)
        with client.websocket_connect('/ws/live') as socket:
            socket.receive_json()
            socket.send_bytes(packet(1, 100))
            self.assertTrue(detector.entered.wait(2))
            for frame_id in (2, 3, 4):
                socket.send_bytes(packet(frame_id, frame_id * 100))
            # Esta respuesta confirma que receive() ya leyo los tres cuadros.
            socket.send_text('{}')
            self.assertEqual(socket.receive_json()['code'], 'invalid_frame')
            detector.resume.set()
            results = [socket.receive_json(), socket.receive_json()]
            self.assertEqual([item['frame_id'] for item in results], [1, 4])
            self.assertEqual(results[-1]['dropped_frames'], 2)

    def test_helmet_alert_saves_correct_frame_and_resets_on_reconnect(self):
        detections = [dict(class_name='moto', confidence=0.91,
                          bbox=[20, 40, 70, 75], is_crash=False),
                      dict(class_name='no_casco', confidence=0.9,
                          bbox=[30, 10, 45, 25], is_crash=False)]
        client = self.client(FakeDetector(detections))
        with patch.object(config, 'AUTO_SAVE_HELMET_SNAPSHOT', True):
            with client.websocket_connect('/ws/live') as socket:
                socket.receive_json()
                socket.send_bytes(packet(9, 100))
                self.assertEqual(socket.receive_json()['type'], 'detections')
                socket.send_bytes(packet(10, 200))
                replies = [socket.receive_json(), socket.receive_json()]
                event = next(item for item in replies if item['type'] == 'event')
                self.assertEqual(event['frame_id'], 10)
                self.assertEqual(event['timestamp_ms'], 200)
                self.assertEqual(event['event'], 'no_casco')
                self.assertEqual(client.get(event['snapshot_url']).status_code, 200)
            with client.websocket_connect('/ws/live') as socket:
                socket.receive_json()
                socket.send_bytes(packet(1, 100))
                self.assertEqual(socket.receive_json()['type'], 'detections')
        self.assertEqual(len(client.get('/events').json()['events']), 1)

    def test_parallel_results_keep_original_client_frame_id(self):
        self.client()
        focus = Mock()
        focus.process.is_alive.return_value = True
        self.runtime.focus = focus
        focus.poll.side_effect = [[dict(frame=3, image=None,
            detections=[dict(class_name='casco', confidence=0.91, bbox=[1, 2, 3, 4])],
            reports=[dict(moto_id=7, violation_confidence=0.9, bbox=[10, 20, 30, 40])])], []]

        async def check():
            session = LiveSession(self.runtime, Mock())
            session.metadata[3] = FramePacket(900, 123.4, b'')
            task = asyncio.create_task(session.poll_helmets())
            detection = await asyncio.wait_for(session.outgoing.get(), 2)
            event = await asyncio.wait_for(session.outgoing.get(), 2)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(detection['type'], 'helmet_detections')
            self.assertEqual(event['type'], 'event')
            self.assertEqual((event['frame_id'], event['timestamp_ms']), (900, 123.4))
            self.assertEqual(event['moto_id'], 7)
        asyncio.run(check())

    def test_api_key_protects_stream_events_and_snapshots(self):
        client = self.client(key='clave-de-prueba')
        self.assertEqual(client.get('/events').status_code, 401)
        self.assertEqual(client.get('/events', headers={'x-api-key': 'clave-de-prueba'}).status_code, 200)
        with self.assertRaises(Exception):
            with client.websocket_connect('/ws/live'):
                pass
        with client.websocket_connect('/ws/live?token=clave-de-prueba') as socket:
            self.assertEqual(socket.receive_json()['type'], 'ready')
        self.assertEqual(client.get('/snapshots/no_casco_' + '0'*32 + '.jpg',
                         headers={'x-api-key': 'clave-de-prueba'}).status_code, 404)

    def test_missing_model_fails_startup(self):
        detector = FakeDetector()
        detector.model_type = 'mock'
        runtime = DetectionRuntime(detector_factory=lambda: detector, data_dir=self.folder)
        with self.assertRaisesRegex(RuntimeError, 'modelo YOLO'):
            with TestClient(create_app(runtime)):
                pass


if __name__ == '__main__':
    unittest.main()
