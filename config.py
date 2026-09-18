"""
Configuración general del detector de choques en video.
Modifica estos valores según el modelo y tus preferencias.
"""
from pathlib import Path

# --- RUTA DEL MODELO ---
# Si descargas los pesos (.pt o .onnx) colócalo en weights/best.pt
MODEL_PATH = "weights/best.pt"

# Soporte opcional para inferencia directa con Roboflow (API)
ROBOFLOW_API_KEY = "ygnvNU2VSJC9icjhS9XH"
ROBOFLOW_MODEL_ID = "accidentes-utn/2"
USE_ROBOFLOW_API = False  # Cambia a True si deseas inferir directamente por API sin descargar el .pt

# --- CLASES DEL MODELO (Roboflow: accidentes-utn v2) ---
ALL_CLASSES = [
    "bus",
    "car",
    "crash",
    "moto",
    "casco",
    "no_casco",
    "truck"
]

# Clases que disparan la ALERTA DE CHOQUE por consola
CRASH_CLASS_NAMES = ["crash"]

# Colores en formato BGR para OpenCV (extraídos del dataset)
CLASS_COLORS = {
    "moto": (235, 183, 0),
    "casco": (0, 200, 0),
    "no_casco": (0, 128, 255),
    "bus": (255, 34, 134),            # #8622FF -> BGR
    "car": (0, 252, 199),            # #C7FC00 -> BGR
    "crash": (206, 255, 0),          # #00FFCE -> BGR (Verde azulado brillante)
    "moto_no_casco": (0, 128, 255),  # #FF8000 -> BGR (Naranja)
    "motocasco": (235, 183, 0),      # #00B7EB -> BGR (Celeste)
    "truck": (86, 0, 254)            # #FE0056 -> BGR (Rojo/Rosa fuerte)
}

# --- CÁMARA WI-FI / IP (STREAM EN VIVO) ---
# URL por defecto para cámara IP / Wi-Fi (RTSP, HTTP o celular)
# Ejemplos:
#   - Celular con IP Webcam: "http://192.168.1.100:8080/video"
#   - Cámara Tapo/Ezviz/Dahua: "rtsp://admin:clave123@192.168.1.50:554/stream1"
DEFAULT_CAMERA_URL = "http://192.168.1.100:8080/video"

# Guardar automáticamente una captura (.jpg) cuando se detecte un choque
AUTO_SAVE_CRASH_SNAPSHOT = True
SNAPSHOTS_DIR = "snapshots_choques"
AUTO_SAVE_HELMET_SNAPSHOT = True
HELMET_SNAPSHOTS_DIR = "snapshots_sin_casco"

# --- PARÁMETROS DE DETECCIÓN ---
# Umbral de confianza mínimo (0.25 = 25% para detectar impactos más sutiles o moderados)
CONFIDENCE_THRESHOLD = 0.25
# Umbrales por clase: la confianza debe ser estrictamente mayor.
CLASS_CONFIDENCE_THRESHOLDS = {
    "car": 0.75,
    "moto": 0.70,
    "casco": 0.40,
    "no_casco": 0.40,
    "crash": 0.75,
}


def passes_detection_threshold(detection):
    name = detection.get("class_name", "").lower()
    confidence = detection.get("confidence", 0.0)
    if name in CLASS_CONFIDENCE_THRESHOLDS:
        return confidence > CLASS_CONFIDENCE_THRESHOLDS[name]
    return confidence >= CONFIDENCE_THRESHOLD


# 416 acelera el modelo en CPU; usa 640 si necesitas mas detalle a distancia.
INFERENCE_IMAGE_SIZE = 416
# Saltar cuadros atrasados de archivos para mantener el ritmo del video.
# Al guardar el video de salida se analizan todos los cuadros.
REALTIME_VIDEO_PLAYBACK = True
INFERENCE_CPU_THREADS = 4
# Segunda pasada sobre recortes originales de motos y ocupantes.
HELMET_FOCUS_ENABLED = True
HELMET_FOCUS_IMAGE_SIZE = 416
HELMET_FOCUS_CPU_THREADS = 2
HELMET_FOCUS_MAX_MOTOS = 3
HELMET_FOCUS_MAX_SAMPLE_GAP = 0.6
# Filtro exclusivo de choques; no afecta la deteccion de vehiculos.
CRASH_CONFIDENCE_THRESHOLD = 0.75
CRASH_CONFIRM_FRAMES = 5       # Minimo de frames positivos consecutivos
CRASH_CONFIRM_SECONDS = 0.5    # Tambien deben persistir este tiempo
CRASH_COOLDOWN_SECONDS = 10.0  # Tiempo minimo entre alertas nuevas
CRASH_CLEAR_SECONDS = 2.0      # Tiempo sin choque para terminar el evento

# Infracciones de casco: reportes por consola, resumen y capturas opcionales.
# Confirmacion breve para motos que cruzan rapido; exige evidencia no_casco.
HELMET_CONFIDENCE_THRESHOLD = CLASS_CONFIDENCE_THRESHOLDS["no_casco"]
HELMET_CONFIRM_FRAMES = 2
HELMET_CONFIRM_SECONDS = 0.08
HELMET_COOLDOWN_SECONDS = CRASH_COOLDOWN_SECONDS
HELMET_CLEAR_SECONDS = CRASH_CLEAR_SECONDS
# Zona de cabezas: hasta 1.5 alturas sobre la moto y su mitad superior.
HELMET_ABOVE_HEIGHT = 1.5
HELMET_SIDE_MARGIN = 0.15      # Margen lateral proporcional al ancho de moto
MOTO_TRACK_MIN_IOU = 0.20     # Solapamiento minimo para seguir la misma moto

# --- VISUALIZACIÓN ---
SHOW_PREVIEW = True          # Mostrar ventana con video en vivo
SHOW_ALL_VEHICLES = True     # Dibujar cajas para todos los vehículos detectados (autos, motos, camiones)
SAVE_OUTPUT_VIDEO = False    # Guardar video con las detecciones
OUTPUT_VIDEO_PATH = "output_analisis.mp4"



