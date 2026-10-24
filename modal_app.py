"""Despliegue serverless del servicio de deteccion en Modal.

No mantiene contenedores calientes: al cerrar la ultima conexion WebSocket el
contenedor se apaga como maximo dos segundos despues.
"""

import os

import modal


APP_DIR = "/root/detection_service"
app = modal.App("deteccion-video-ia")

# Se empaqueta el peso dentro de la imagen. Asi no hace falta un Volume (que
# podria agregar almacenamiento facturable) y cada contenedor es autocontenido.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "anyio>=4,<5",
        "fastapi>=0.115,<1",
        "opencv-python-headless>=4.8.0",
        "ultralytics>=8.0.0",
        "uvicorn>=0.30,<1",
        "websockets>=14,<17",
    )
    .add_local_file("config.py", f"{APP_DIR}/config.py", copy=True)
    .add_local_file("crash_filter.py", f"{APP_DIR}/crash_filter.py", copy=True)
    .add_local_file("detector.py", f"{APP_DIR}/detector.py", copy=True)
    .add_local_file("helmet_focus.py", f"{APP_DIR}/helmet_focus.py", copy=True)
    .add_local_file("helmet_monitor.py", f"{APP_DIR}/helmet_monitor.py", copy=True)
    .add_local_file("service_api.py", f"{APP_DIR}/service_api.py", copy=True)
    .add_local_file("weights/best.pt", f"{APP_DIR}/weights/best.pt", copy=True)
    .env({"PYTHONPATH": APP_DIR, "DETECTION_MAX_CONNECTIONS": "2"})
)


@app.function(
    image=image,
    gpu="L4",
    # Cero instancias y cero buffer: ninguna GPU queda facturando en reposo.
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=2,
    # Una camara puede permanecer conectada por horas.
    timeout=86_400,
    secrets=[modal.Secret.from_name("detection-service")],
)
@modal.concurrent(max_inputs=2)
@modal.asgi_app()
def detection_api():
    """Expone FastAPI y hasta dos WebSockets en un unico contenedor GPU."""
    from service_api import create_app

    api_key = os.environ.get("DETECTION_API_KEY", "")
    if not api_key:
        raise RuntimeError("Falta DETECTION_API_KEY en el secreto detection-service")
    return create_app(api_key=api_key, max_connections=2)
