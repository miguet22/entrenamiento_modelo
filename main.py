"""
Programa principal para análisis de video y detección de choques mediante IA.
Soporta:
  1. Archivos de video locales (Buscador de Microsoft)
  2. Cámaras Wi-Fi / IP en vivo (RTSP / HTTP)
"""
import os
import sys
import time
from datetime import datetime
import cv2
from colorama import init, Fore, Style

import config
from file_picker import select_video_file
from detector import CrashDetector

# Inicializar colorama para colores en la consola
init(autoreset=True)

def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato MM:SS.ms"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"

def print_banner():
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)
    print(Fore.YELLOW + Style.BRIGHT + "       SISTEMA DE DETECCIÓN DE CHOQUES EN VIDEO CON IA       ")
    print(Fore.LIGHTBLACK_EX + f"       Proyecto: {config.ROBOFLOW_MODEL_ID} | Umbral: {config.CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(Fore.LIGHTBLACK_EX + "       Clases: " + ", ".join(config.ALL_CLASSES))
    print(Fore.CYAN + Style.BRIGHT + "=" * 70 + "\n")

def get_source_selection():
    """Muestra el menú de selección de fuente de video."""
    print(Fore.WHITE + Style.BRIGHT + "Selecciona el origen del video:")
    print(Fore.GREEN + "  [1] " + Fore.WHITE + "Cargar archivo de video (Explorador de Windows)")
    print(Fore.CYAN + "  [2] " + Fore.WHITE + "Conectar a Cámara Wi-Fi / IP en Vivo (RTSP / HTTP)")
    print(Fore.RED + "  [0] " + Fore.LIGHTBLACK_EX + "Salir\n")

    while True:
        choice = input(Fore.YELLOW + "Elige una opción [1/2/0]: " + Fore.RESET).strip()
        if choice in ["1", "2", "0"]:
            return choice
        print(Fore.RED + "Opción inválida. Ingresa 1, 2 o 0.")

def main():
    print_banner()

    choice = get_source_selection()
    if choice == "0":
        print(Fore.YELLOW + "\nSaliendo del programa...")
        return

    is_live_stream = False
    source_name = ""
    video_source = None

    if choice == "1":
        print(Fore.WHITE + "\n[1/3] Abriendo explorador de archivos para seleccionar video...")
        video_source = select_video_file()
        if not video_source:
            print(Fore.RED + "\n[!] No se seleccionó ningún archivo de video. Proceso cancelado.")
            return
        source_name = os.path.basename(video_source)
        print(Fore.GREEN + f"[✓] Video seleccionado: {source_name}")

    elif choice == "2":
        is_live_stream = True
        print(Fore.CYAN + "\n[1/3] Configurar Cámara Wi-Fi / IP")
        print(Fore.LIGHTBLACK_EX + "  Ejemplos:")
        print(Fore.LIGHTBLACK_EX + "  • Celular (IP Webcam): http://192.168.1.100:8080/video")
        print(Fore.LIGHTBLACK_EX + "  • Cámara IP / RTSP:    rtsp://admin:12345@192.168.1.50:554/stream1")
        print(Fore.LIGHTBLACK_EX + f"  • Por defecto:         {config.DEFAULT_CAMERA_URL}")
        
        user_url = input(Fore.YELLOW + f"\nIngresa la URL del stream [Presiona Enter para '{config.DEFAULT_CAMERA_URL}']: " + Fore.RESET).strip()
        video_source = user_url if user_url else config.DEFAULT_CAMERA_URL
        source_name = f"Stream en Vivo ({video_source})"
        print(Fore.GREEN + f"[✓] Conectando a stream: {video_source}")

    # 2. Inicializar el detector de IA
    print(Fore.WHITE + "\n[2/3] Inicializando modelo de IA...")
    detector = CrashDetector(
        model_path=config.MODEL_PATH,
        crash_classes=config.CRASH_CLASS_NAMES,
        conf_threshold=config.CONFIDENCE_THRESHOLD,
        use_roboflow_api=config.USE_ROBOFLOW_API
    )

    # 3. Abrir la fuente de video con OpenCV
    if is_live_stream:
        # Optimización de buffer para streams RTSP/HTTP en vivo
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;500000"
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(Fore.RED + f"\n[ERROR] No se pudo conectar o abrir la fuente de video: {video_source}")
        if is_live_stream:
            print(Fore.YELLOW + "[TIP] Verifica que el dispositivo esté en la misma red Wi-Fi y que la IP/puerto sean correctos.")
        return

    # Si es stream en vivo, crear carpeta de snapshots si está activo
    if config.AUTO_SAVE_CRASH_SNAPSHOT:
        os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live_stream else 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    duration_sec = total_frames / fps if total_frames > 0 else 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(Fore.WHITE + f"\n[3/3] Información de la Fuente:")
    print(Fore.LIGHTCYAN_EX + f"    • Origen     : {source_name}")
    print(Fore.LIGHTCYAN_EX + f"    • Resolución : {width}x{height}")
    print(Fore.LIGHTCYAN_EX + f"    • FPS        : {fps:.2f}")
    if not is_live_stream:
        print(Fore.LIGHTCYAN_EX + f"    • Duración   : {format_timestamp(duration_sec)} ({total_frames} frames)")
    else:
        print(Fore.GREEN + "    • Modo       : TRANSMISIÓN EN VIVO CONTINUA (Wi-Fi/IP)")
    
    print(Fore.CYAN + "-" * 70)
    print(Fore.YELLOW + "Iniciando análisis... (Presiona 'Q' en la ventana del video para detener)\n")

    # Configuración de salida de video opcional
    out_writer = None
    if config.SAVE_OUTPUT_VIDEO and not is_live_stream:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # Control de eventos detectados
    crash_events = []
    current_event = None
    frame_idx = 0
    last_crash_frame = -9999
    start_time_real = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_live_stream:
                    print(Fore.YELLOW + "\n[!] Pérdida de señal con la cámara Wi-Fi. Reintentando...")
                    time.sleep(1)
                    continue
                else:
                    break

            frame_idx += 1
            if not is_live_stream:
                current_time_sec = frame_idx / fps
                time_str = format_timestamp(current_time_sec)
            else:
                now = datetime.now()
                time_str = now.strftime("%H:%M:%S")

            # Inferencia con IA
            detections = detector.predict_frame(frame)

            # Filtrar detecciones de choque
            crash_dets = [d for d in detections if d.get("is_crash", False)]
            has_crash = len(crash_dets) > 0

            if has_crash:
                max_conf = max(d["confidence"] for d in crash_dets)
                class_name = crash_dets[0]["class_name"]

                # Loguear en consola con color llamativo
                frame_info = f"{frame_idx}/{total_frames}" if not is_live_stream else f"Frame #{frame_idx}"
                print(
                    Fore.RED + Style.BRIGHT + f"[🚨 CHOQUE DETECTADO] " +
                    Fore.YELLOW + f"Hora/Tiempo: {time_str} " +
                    Fore.WHITE + f"| {frame_info} " +
                    Fore.GREEN + f"| Confianza: {max_conf * 100:.1f}% " +
                    Fore.MAGENTA + f"| Clase: '{class_name}'"
                )

                # Guardar captura automática si es stream o video
                if config.AUTO_SAVE_CRASH_SNAPSHOT and (frame_idx - last_crash_frame > config.DEBOUNCE_FRAMES):
                    snap_name = f"choque_{datetime.now().strftime('%Y%m%d_%H%M%S')}_f{frame_idx}.jpg"
                    snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)
                    cv2.imwrite(snap_path, frame)
                    print(Fore.LIGHTGREEN_EX + f"    📸 Captura guardada en: {snap_path}")

                # Agrupamiento de eventos (debounce)
                if frame_idx - last_crash_frame > config.DEBOUNCE_FRAMES:
                    if current_event:
                        crash_events.append(current_event)
                    current_event = {
                        "start_time": time_str,
                        "end_time": time_str,
                        "start_frame": frame_idx,
                        "end_frame": frame_idx,
                        "max_conf": max_conf,
                        "class_name": class_name
                    }
                else:
                    if current_event:
                        current_event["end_time"] = time_str
                        current_event["end_frame"] = frame_idx
                        current_event["max_conf"] = max(current_event["max_conf"], max_conf)

                last_crash_frame = frame_idx

            # Dibujar en el frame
            if config.SHOW_PREVIEW or config.SAVE_OUTPUT_VIDEO:
                annotated_frame = frame.copy()

                # Dibujar todas las cajas de objetos
                for d in detections:
                    if not config.SHOW_ALL_VEHICLES and not d.get("is_crash", False):
                        continue

                    c_name = d["class_name"]
                    color = config.CLASS_COLORS.get(c_name, (0, 255, 0))

                    if "bbox" in d:
                        x1, y1, x2, y2 = d["bbox"]
                        thickness = 3 if d.get("is_crash", False) else 2
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        label = f"{c_name.upper()} {d['confidence']*100:.0f}%"
                        cv2.putText(annotated_frame, label, (x1, max(18, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Banner superior si hay alerta
                if has_crash:
                    cv2.rectangle(annotated_frame, (0, 0), (width, 42), (0, 0, 220), -1)
                    cv2.putText(annotated_frame, f"¡ALERTA: CHOQUE DETECTADO! [{time_str}]", (20, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                else:
                    # Mostrar barra de tiempo / estado superior
                    cv2.rectangle(annotated_frame, (0, 0), (350, 36), (0, 0, 0), -1)
                    mode_tag = "🔴 EN VIVO" if is_live_stream else "REPRODUCIENDO"
                    cv2.putText(annotated_frame, f"{mode_tag} | {time_str}", (15, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if out_writer:
                    out_writer.write(annotated_frame)

                if config.SHOW_PREVIEW:
                    display_frame = annotated_frame
                    if width > 1280 or height > 720:
                        scale = min(1280 / width, 720 / height)
                        display_frame = cv2.resize(annotated_frame, (int(width * scale), int(height * scale)))

                    win_title = "Detector de Choques - EN VIVO (Wi-Fi)" if is_live_stream else "Detector de Choques - Video"
                    cv2.imshow(win_title, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(Fore.YELLOW + "\n[!] Análisis detenido por el usuario.")
                        break

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()

    if current_event:
        crash_events.append(current_event)

    # --- RESUMEN FINAL ---
    print("\n" + Fore.CYAN + Style.BRIGHT + "=" * 70)
    print(Fore.CYAN + Style.BRIGHT + "                     RESUMEN DE ANÁLISIS                      ")
    print(Fore.CYAN + Style.BRIGHT + "=" * 70)
    print(Fore.WHITE + f"Total de frames analizados: {frame_idx}")
    print(Fore.WHITE + f"Total de eventos de choque detectados: {len(crash_events)}")

    if crash_events:
        print(Fore.YELLOW + "\nDetalle de eventos de impacto:")
        for idx, ev in enumerate(crash_events, 1):
            print(
                Fore.LIGHTRED_EX + f"  • Evento #{idx}: " +
                Fore.WHITE + f"Desde {ev['start_time']} hasta {ev['end_time']} " +
                Fore.GREEN + f"(Pico Confianza: {ev['max_conf'] * 100:.1f}%) " +
                Fore.LIGHTBLACK_EX + f"[Frames {ev['start_frame']} - {ev['end_frame']}]"
            )
    else:
        print(Fore.GREEN + "\nNo se detectaron choques en la sesión.")

    print(Fore.CYAN + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
