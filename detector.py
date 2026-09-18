"""
Módulo de inferencia y detección con IA.
Compatible con modelos YOLO (.pt), Roboflow Inference API, ONNX y clasificadores.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
import config

class CrashDetector:
    def __init__(self, model_path: str = config.MODEL_PATH,
                 crash_classes: List[str] = config.CRASH_CLASS_NAMES,
                 conf_threshold: float = config.CONFIDENCE_THRESHOLD,
                 use_roboflow_api: bool = config.USE_ROBOFLOW_API,
                 image_size: int = config.INFERENCE_IMAGE_SIZE,
                 cpu_threads: int = config.INFERENCE_CPU_THREADS):
        path = Path(model_path)
        self.model_path = str(path if path.is_absolute()
                              else Path(config.__file__).resolve().parent / path)
        self.crash_classes = [c.lower() for c in crash_classes]
        self.conf_threshold = conf_threshold
        self.use_roboflow_api = use_roboflow_api
        self.model = None
        self.model_type = "mock"
        self.rf_client = None
        self.image_size = image_size
        self.cpu_threads = cpu_threads

        self._init_detector()

    def _init_detector(self):
        """Inicializa el modelo local o el cliente de API de Roboflow."""
        if self.use_roboflow_api:
            try:
                from inference_sdk import InferenceHTTPClient
                self.rf_client = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key=config.ROBOFLOW_API_KEY
                )
                self.model_type = "roboflow_api"
                print(f"[OK] Conectado a Roboflow API (Modelo: {config.ROBOFLOW_MODEL_ID})")
                return
            except ImportError:
                print("[AVISO] Para usar Roboflow API instala: pip install inference-sdk")
                print("         Intentando cargar modelo local...")

        # Cargar modelo local
        if not os.path.exists(self.model_path):
            print(f"\n[AVISO] No se encontró el archivo de modelo local en '{self.model_path}'.")
            print("        - Si estás entrenando en Roboflow, puedes descargar el archivo 'best.pt' y colocarlo en 'weights/best.pt'.")
            print("        - O puedes activar 'USE_ROBOFLOW_API = True' en config.py para usar la API directa.")
            print("[INFO] El programa funcionará en modo de espera hasta que el modelo esté disponible.\n")
            self.model_type = "mock"
            return

        # Intentar cargar con Ultralytics (YOLO)
        try:
            from ultralytics import YOLO
            import torch
            if not torch.cuda.is_available():
                torch.set_num_threads(self.cpu_threads)
            self.model = YOLO(self.model_path)
            self.model_type = "yolo"
            print(f"[OK] Modelo YOLO cargado con éxito desde: {self.model_path}")
            if hasattr(self.model, 'names'):
                print(f"[INFO] Clases del modelo: {self.model.names}")
            return
        except ImportError:
            print("[ERROR] Ultralytics no está instalado. Ejecuta: pip install ultralytics")
        except Exception as e:
            print(f"[ERROR] Error al cargar modelo YOLO: {e}")

        # Intentar cargar con OpenCV DNN (para .onnx)
        if self.model_path.endswith(".onnx"):
            try:
                self.model = cv2.dnn.readNetFromONNX(self.model_path)
                self.model_type = "onnx"
                print(f"[OK] Modelo ONNX cargado con OpenCV DNN: {self.model_path}")
                return
            except Exception as e:
                print(f"[ERROR] Error al cargar ONNX: {e}")

        self.model_type = "mock"

    def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Ejecuta la inferencia sobre un frame (BGR).
        Retorna lista de detecciones:
        [
            {
                "class_name": "crash",
                "confidence": 0.92,
                "bbox": [x1, y1, x2, y2],
                "is_crash": True
            }
        ]
        """
        detections = []

        if self.model_type == "yolo" and self.model is not None:
            results = self.model(frame, conf=self.conf_threshold,
                                 imgsz=self.image_size, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = self.model.names[cls_id].lower() if hasattr(self.model, 'names') else str(cls_id)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    is_crash = any(target in class_name for target in self.crash_classes)
                    detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "is_crash": is_crash
                    })

        elif self.model_type == "roboflow_api" and self.rf_client is not None:
            try:
                # Inferencia vía Roboflow Inference API
                res = self.rf_client.infer(frame, model_id=config.ROBOFLOW_MODEL_ID)
                for pred in res.get("predictions", []):
                    conf = float(pred.get("confidence", 0.0))
                    if conf >= self.conf_threshold:
                        class_name = pred.get("class", "").lower()
                        x = pred.get("x", 0)
                        y = pred.get("y", 0)
                        w = pred.get("width", 0)
                        h = pred.get("height", 0)
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)

                        is_crash = any(target in class_name for target in self.crash_classes)
                        detections.append({
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "is_crash": is_crash
                        })
            except Exception as e:
                pass

        return [d for d in detections if config.passes_detection_threshold(d)]
