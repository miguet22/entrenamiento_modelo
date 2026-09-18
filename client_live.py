"""Cliente de ejemplo: transmite camara/video y recibe JSON simultaneamente."""

import argparse
import asyncio
import json
from pathlib import Path
import struct
import time

import cv2
from websockets.asyncio.client import connect

from live_capture import LatestFrameCapture


def encode_packet(frame, frame_id, timestamp_ms):
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('No se pudo codificar el cuadro')
    metadata = json.dumps(dict(frame_id=frame_id, timestamp_ms=timestamp_ms)).encode('utf-8')
    return struct.pack('!I', len(metadata)) + metadata + encoded.tobytes()


async def transmit(args):
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('No se pudo abrir la camara o video')
    is_file = isinstance(source, str) and Path(source).is_file()
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    if not 0 < source_fps <= 240:
        source_fps = 30
    if not is_file:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture = LatestFrameCapture(capture)
    frame_index = 0

    def read_sample(elapsed):
        nonlocal frame_index
        if is_file:
            target = int(elapsed * source_fps)
            while frame_index < target:
                if not capture.grab():
                    return False, None
                frame_index += 1
        result = capture.read()
        frame_index += 1
        return result

    async def receive_results(socket):
        async for message in socket:
            result = json.loads(message)
            if result['type'] in ('event', 'error'):
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(result['type'], 'cuadro', result.get('frame_id'),
                      'detecciones', len(result.get('detections', [])))

    headers = {'x-api-key': args.api_key} if args.api_key else None
    try:
        async with connect(args.url, additional_headers=headers, max_size=6*1024*1024) as socket:
            ready = json.loads(await socket.recv())
            print(json.dumps(ready, ensure_ascii=False))
            if ready.get('type') != 'ready':
                return
            receiver = asyncio.create_task(receive_results(socket))
            try:
                start = time.monotonic()
                sequence = 0
                period = 1 / args.fps
                while not receiver.done():
                    tick = time.monotonic()
                    ok, frame = await asyncio.to_thread(read_sample, tick - start)
                    if not ok:
                        break
                    sequence += 1
                    packet = await asyncio.to_thread(encode_packet, frame, sequence,
                                                     (tick - start) * 1000)
                    await socket.send(packet)
                    await asyncio.sleep(max(0, period - (time.monotonic() - tick)))
                await asyncio.sleep(0.5)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    finally:
        capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='0 para webcam, ruta de video o URL RTSP/HTTP')
    parser.add_argument('--url', default='ws://127.0.0.1:8001/ws/live')
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--api-key', default='')
    args = parser.parse_args()
    if not 0 < args.fps <= 30:
        parser.error('--fps debe estar entre 0 y 30')
    try:
        asyncio.run(transmit(args))
    except KeyboardInterrupt:
        pass
