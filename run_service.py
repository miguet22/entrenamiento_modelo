"""Servidor en consola; el servicio Windows reutiliza la misma configuracion."""

import multiprocessing
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_server():
    import uvicorn
    from service_api import MAX_PACKET_BYTES, create_app
    settings_path = ROOT / 'service_settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8-sig')) if settings_path.exists() else {}
    host = os.environ.get('DETECTION_HOST', settings.get('host', '127.0.0.1'))
    port = int(os.environ.get('DETECTION_PORT', settings.get('port', 8001)))
    api_key = os.environ.get('DETECTION_API_KEY', settings.get('api_key', ''))
    if not 1 <= port <= 65535:
        raise ValueError('Puerto fuera de rango')
    if host not in ('127.0.0.1', 'localhost', '::1') and not api_key:
        raise RuntimeError('Configura DETECTION_API_KEY para escuchar fuera de localhost')
    app = create_app(api_key=api_key)
    server = uvicorn.Server(uvicorn.Config(
        app, host=host, port=port, workers=1,
        ws_max_size=MAX_PACKET_BYTES, ws_max_queue=1,
        timeout_graceful_shutdown=5))
    app.state.runtime.on_fatal = lambda: setattr(server, 'should_exit', True)
    return server


if __name__ == '__main__':
    multiprocessing.freeze_support()
    os.chdir(ROOT)
    server = build_server()
    server.run()
    if server.config.app.state.runtime.fatal_error:
        raise SystemExit(1)
