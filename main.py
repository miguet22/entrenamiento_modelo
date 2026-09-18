"""
Programa principal para análisis de video y detección de choques mediante IA.
Soporta:
  1. Archivos de video locales (Buscador de Microsoft)
  2. Cámaras Wi-Fi / IP en vivo (RTSP / HTTP)
  3. Cámara integrada o USB de la PC
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
from crash_filter import CrashFilter
from helmet_monitor import HelmetMonitor
from live_capture import LatestFrameCapture
from helmet_focus import HelmetFocus

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
    print(Fore.LIGHTBLACK_EX + f"       Proyecto: {config.ROBOFLOW_MODEL_ID}")
    print(Fore.LIGHTBLACK_EX + "       Umbrales: auto >75%, moto >70%, casco >50%, sin casco >50%, choque >60%")
    print(Fore.LIGHTBLACK_EX + "       Clases: " + ", ".join(config.ALL_CLASSES))
    print(Fore.CYAN + Style.BRIGHT + "=" * 70 + "\n")

def get_source_selection():
    """Muestra el menú de selección de fuente de video."""
    print(Fore.WHITE + Style.BRIGHT + "Selecciona el origen del video:")
    print(Fore.GREEN + "  [1] " + Fore.WHITE + "Cargar archivo de video (Explorador de Windows)")
    print(Fore.CYAN + "  [2] " + Fore.WHITE + "Conectar a Cámara Wi-Fi / IP en Vivo (RTSP / HTTP)")
    print(Fore.CYAN + "  [3] " + Fore.WHITE + "Usar cámara de mi PC (integrada / USB)")
    print(Fore.RED + "  [0] " + Fore.LIGHTBLACK_EX + "Salir\n")

    while True:
        choice = input(Fore.YELLOW + "Elige una opción [1/2/3/0]: " + Fore.RESET).strip()
        if choice in ["1", "2", "3", "0"]:
            return choice
        print(Fore.RED + "Opción inválida. Ingresa 1, 2, 3 o 0.")

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

    elif choice == "3":
        is_live_stream = True
        print(Fore.CYAN + "\n[1/3] Configurar cámara de la PC")
        print(Fore.LIGHTBLACK_EX + "  Usa 0 para la cámara principal; prueba 1 o 2 si tienes otra cámara USB.")
        while True:
            camera_index = input(Fore.YELLOW + "Número de cámara [Enter para 0]: " + Fore.RESET).strip()
            try:
                video_source = int(camera_index or "0")
                if video_source >= 0:
                    break
            except ValueError:
                pass
            print(Fore.RED + "Ingresa un número entero mayor o igual a 0.")
        source_name = f"Cámara de la PC ({video_source})"

    # 2. Inicializar el detector de IA
    print(Fore.WHITE + "\n[2/3] Inicializando modelo de IA...")
    detector = CrashDetector(
        model_path=config.MODEL_PATH,
        crash_classes=config.CRASH_CLASS_NAMES,
        conf_threshold=config.CONFIDENCE_THRESHOLD,
        use_roboflow_api=config.USE_ROBOFLOW_API
    )

    # 3. Abrir la fuente de video con OpenCV
    if choice == "2":
        # Optimización de buffer para streams RTSP/HTTP en vivo
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;500000"
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        cap.release()
        print(Fore.RED + f"\n[ERROR] No se pudo conectar o abrir la fuente de video: {video_source}")
        if choice == "3":
            print(Fore.YELLOW + "[TIP] Habilita el acceso a la cámara para aplicaciones de escritorio en Windows, cierra otras apps que la usen o prueba otro número de cámara.")
        elif choice == "2":
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
    if is_live_stream:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap = LatestFrameCapture(cap)

    print(Fore.WHITE + f"\n[3/3] Información de la Fuente:")
    print(Fore.LIGHTCYAN_EX + f"    • Origen     : {source_name}")
    print(Fore.LIGHTCYAN_EX + f"    • Resolución : {width}x{height}")
    print(Fore.LIGHTCYAN_EX + f"    • FPS        : {fps:.2f}")
    if not is_live_stream:
        print(Fore.LIGHTCYAN_EX + f"    • Duración   : {format_timestamp(duration_sec)} ({total_frames} frames)")
    else:
        camera_type = "PC / USB" if choice == "3" else "Wi-Fi/IP"
        print(Fore.GREEN + f"    • Modo       : TRANSMISIÓN EN VIVO CONTINUA ({camera_type})")
    
    print(Fore.CYAN + "-" * 70)
    print(Fore.YELLOW + "Iniciando análisis... (Presiona 'Q' en la ventana del video para detener)\n")

    # Configuración de salida de video opcional
    out_writer = None
    if config.SAVE_OUTPUT_VIDEO and not is_live_stream:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # Control de eventos detectados
    crash_events = []
    helmet_events = []
    current_event = None
    frame_idx = 0
    crash_filter = CrashFilter(
        min_frames=config.CRASH_CONFIRM_FRAMES,
        confirm_seconds=config.CRASH_CONFIRM_SECONDS,
        cooldown_seconds=config.CRASH_COOLDOWN_SECONDS,
        clear_seconds=config.CRASH_CLEAR_SECONDS,
    )
    helmet_monitor = HelmetMonitor(
        confidence=config.HELMET_CONFIDENCE_THRESHOLD,
        min_frames=config.HELMET_CONFIRM_FRAMES,
        confirm_seconds=config.HELMET_CONFIRM_SECONDS,
        cooldown_seconds=config.HELMET_COOLDOWN_SECONDS,
        clear_seconds=config.HELMET_CLEAR_SECONDS,
        above_height=config.HELMET_ABOVE_HEIGHT,
        side_margin=config.HELMET_SIDE_MARGIN,
        min_iou=config.MOTO_TRACK_MIN_IOU,
    )

    def handle_helmet_reports(reports, frame, time_str, frame_idx):
        for report in reports:
            helmet_events.append({
                "time": time_str,
                "frame": frame_idx,
                "moto_id": report["moto_id"],
                "confidence": report["violation_confidence"],
            })
            print(
                Fore.YELLOW + Style.BRIGHT + "[INFRACCION: SIN CASCO] " +
                Fore.WHITE + f"Hora/Tiempo: {time_str} | Frame #{frame_idx} " +
                f"| Moto #{report['moto_id']} | Al menos un ocupante sin casco " +
                Fore.GREEN + f"| Confianza: {report['violation_confidence'] * 100:.1f}%"
            )
            if config.AUTO_SAVE_HELMET_SNAPSHOT:
                try:
                    os.makedirs(config.HELMET_SNAPSHOTS_DIR, exist_ok=True)
                    snap_name = (
                        f"sin_casco_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                        f"_moto{report['moto_id']}_f{frame_idx}.jpg"
                    )
                    snap_path = os.path.join(config.HELMET_SNAPSHOTS_DIR, snap_name)
                    if not cv2.imwrite(snap_path, frame):
                        raise OSError("No se pudo escribir la imagen")
                    print(Fore.LIGHTGREEN_EX + f"    Captura guardada en: {snap_path}")
                except (OSError, cv2.error) as exc:
                    print(Fore.RED + f"    [ERROR] No se pudo guardar la captura sin casco: {exc}")

    helmet_focus = None
    if config.HELMET_FOCUS_ENABLED and detector.model_type == 'yolo':
        print(Fore.CYAN + '[CASCO] Preparando analisis paralelo de motos y ocupantes...')
        try:
            helmet_focus = HelmetFocus()
        except Exception as exc:
            print(Fore.YELLOW + f'[CASCO] Usando deteccion general: {exc}')
        else:
            print(Fore.CYAN + '[CASCO] Analisis paralelo activado.')

    focus_preview = None
    focus_preview_time = 0.0

    def handle_focus_results(samples):
        nonlocal helmet_focus, focus_preview, focus_preview_time
        for sample in samples:
            if 'error' in sample:
                print(Fore.YELLOW + f"[CASCO] Enfoque no disponible: {sample['error']}. Usando deteccion general.")
                helmet_focus.close()
                helmet_focus = None
                break
            focus_preview = sample['preview']
            focus_preview_time = time.monotonic()
            handle_helmet_reports(sample['reports'], sample['image'],
                                  sample['time'], sample['frame'])

    realtime_playback = (not is_live_stream and config.REALTIME_VIDEO_PLAYBACK
                         and not config.SAVE_OUTPUT_VIDEO)
    playback_start = time.monotonic()
    analyzed_frames = 0
    skipped_frames = 0
    try:
        while True:
            if realtime_playback:
                # Descartar cuadros vencidos sin ejecutar IA sobre ellos.
                target_frame = int((time.monotonic() - playback_start) * fps)
                while frame_idx < target_frame:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    skipped_frames += 1
                delay = frame_idx / fps - (time.monotonic() - playback_start)
                if delay > 0:
                    time.sleep(delay)
            ret, frame = cap.read()
            if not ret:
                if choice == "3":
                    print(Fore.RED + "\n[!] No se pudo obtener imagen de la cámara de la PC. Revisa la conexión y los permisos, o prueba otro número de cámara.")
                    break
                if is_live_stream:
                    print(Fore.YELLOW + "\n[!] Pérdida de señal con la cámara Wi-Fi. Reintentando...")
                    crash_filter.reset_candidate()
                    helmet_monitor.reset_candidates()
                    if helmet_focus is not None:
                        helmet_focus.reset()
                    time.sleep(1)
                    continue
                else:
                    break

            frame_idx += 1
            analyzed_frames += 1
            if not is_live_stream:
                current_time_sec = frame_idx / fps
                time_str = format_timestamp(current_time_sec)
            else:
                now = datetime.now()
                time_str = now.strftime("%H:%M:%S")

            # Inferencia con IA
            detections = [d for d in detector.predict_frame(frame)
                          if config.passes_detection_threshold(d)]

            # Filtrar detecciones de choque
            crash_dets = [d for d in detections if d.get("is_crash", False)
                          and d["confidence"] >= config.CRASH_CONFIDENCE_THRESHOLD]
            has_crash = len(crash_dets) > 0

            event_time = time.monotonic() if is_live_stream else current_time_sec
            if helmet_focus is not None:
                helmet_focus.submit(frame, detections, event_time, time_str, frame_idx)
                handle_focus_results(helmet_focus.poll())
            else:
                handle_helmet_reports(helmet_monitor.update(detections, event_time),
                                      frame, time_str, frame_idx)
            new_event = crash_filter.update(has_crash, event_time)
            confirmed_crash = has_crash and crash_filter.active

            if confirmed_crash and not new_event and current_event:
                current_event["end_time"] = time_str
                current_event["end_frame"] = frame_idx
                current_event["max_conf"] = max(
                    current_event["max_conf"], max(d["confidence"] for d in crash_dets))

            if new_event:
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
                if config.AUTO_SAVE_CRASH_SNAPSHOT:
                    snap_name = f"choque_{datetime.now().strftime('%Y%m%d_%H%M%S')}_f{frame_idx}.jpg"
                    snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)
                    cv2.imwrite(snap_path, frame)
                    print(Fore.LIGHTGREEN_EX + f"    📸 Captura guardada en: {snap_path}")

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

            # Dibujar en el frame
            if config.SHOW_PREVIEW or config.SAVE_OUTPUT_VIDEO:
                annotated_frame = frame.copy()

                # Dibujar todas las cajas de objetos
                for d in detections:
                    if d.get("is_crash", False) and (
                            not confirmed_crash
                            or d["confidence"] < config.CRASH_CONFIDENCE_THRESHOLD):
                        continue
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
                if confirmed_crash:
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
                    if (focus_preview is not None
                            and time.monotonic() - focus_preview_time < 0.6):
                        ph, pw = focus_preview.shape[:2]
                        scale = min(240 / pw, 200 / ph,
                                    annotated_frame.shape[1] / pw,
                                    annotated_frame.shape[0] / ph)
                        zoom = cv2.resize(focus_preview,
                                          (max(1, int(pw * scale)),
                                           max(1, int(ph * scale))))
                        zh, zw = zoom.shape[:2]
                        annotated_frame[:zh, -zw:] = zoom
                        cv2.putText(annotated_frame, 'ZOOM MOTO',
                                    (annotated_frame.shape[1] - zw + 4, 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 255, 255), 1)
                    win_title = f"Detector de Choques - EN VIVO ({camera_type})" if is_live_stream else "Detector de Choques - Video"
                    if analyzed_frames == 1:
                        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(win_title, 1600, 900)
                    cv2.imshow(win_title, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(Fore.YELLOW + "\n[!] Análisis detenido por el usuario.")
                        break

    finally:
        if helmet_focus is not None:
            try:
                handle_focus_results(helmet_focus.finish())
            finally:
                if helmet_focus is not None:
                    helmet_focus.close()
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
    print(Fore.WHITE + f"Total de frames analizados: {analyzed_frames}")
    if skipped_frames:
        print(Fore.WHITE + f"Frames omitidos para mantener tiempo real: {skipped_frames}")
    print(Fore.WHITE + f"Total de eventos de choque detectados: {len(crash_events)}")
    print(Fore.WHITE + f"Total de infracciones por falta de casco: {len(helmet_events)}")
    for ev in helmet_events:
        print(Fore.YELLOW + f"  Moto #{ev['moto_id']}: ocupante sin casco "
              + f"| {ev['time']} | Frame {ev['frame']} "
              + f"| Confianza: {ev['confidence'] * 100:.1f}%")

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
