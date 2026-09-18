# Servicio de deteccion en vivo

El servidor carga YOLO y el proceso paralelo de cascos una vez al iniciar.
Funciona sin ventanas, explorador de archivos ni preguntas por consola.
La app cliente captura el video, envia cuadros y recibe detecciones y eventos
por un WebSocket. Esta primera version admite **una transmision activa**.
Otra conexion recibe `busy`; puede consultar `/health` y `/events` por HTTP.

## Preparacion e inicio

Desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-service.txt
.\.venv\Scripts\python.exe run_service.py
```

Tambien podes abrir `iniciar_servicio.bat`. Espera a que aparezca
`Application startup complete`. Consulta `http://127.0.0.1:8001/health`:
`status: ready` indica que ambos modelos estan preparados.
La documentacion HTTP esta en `http://127.0.0.1:8001/docs`.
El protocolo WebSocket esta documentado abajo.

## Instalacion como servicio Windows

Cierra primero el servidor de consola para liberar el puerto 8001.
Ejecuta **instalar_servicio.bat como administrador**. Instala e inicia
`DeteccionVideoIA` con inicio automatico diferido y reinicio ante fallos.
No requiere mantener una consola abierta ni abrir sesion de usuario.

Comandos en una terminal de administrador:

```powershell
.\.venv\Scripts\python.exe windows_service.py status
.\.venv\Scripts\python.exe windows_service.py stop
.\.venv\Scripts\python.exe windows_service.py start
.\.venv\Scripts\python.exe windows_service.py remove
```

Logs: `service_data/service.log`. Capturas: `service_data/snapshots/`.
Mantene la carpeta del proyecto y `.venv` en su ubicacion actual mientras
el servicio este instalado. Para moverlos, desinstala y reinstala.

Los umbrales, la confirmacion temporal y la resolucion siguen en `config.py`.
Se preservan sus valores actuales; reinicia el servicio luego de cambiarlos.
La ventana del programa original sigue disponible con `iniciar.bat`.

## Consumir desde otra app

Conexion: `ws://127.0.0.1:8001/ws/live`. Al conectar recibis:

```json
{"type":"ready","stream_id":"...","protocol_version":1,"recommended_fps":10,"helmet_focus":true}
```

Envia aproximadamente 8-10 cuadros por segundo, **sin esperar una respuesta
para enviar el siguiente**. Mostra el video en tu app independientemente.
Cada cuadro debe tener `frame_id` entero y `timestamp_ms` numerico, ambos
estrictamente crecientes dentro de la conexion. Usa el tiempo de captura,
por ejemplo milisegundos desde el comienzo de la transmision.

La opcion sencilla es enviar un mensaje JSON por cuadro:

```json
{"type":"frame","frame_id":1,"timestamp_ms":100,"image_base64":"JPEG_EN_BASE64"}
```

La opcion recomendada evita base64: un mensaje binario atomico con:

1. 4 bytes: longitud del JSON de metadata, entero unsigned big-endian.
2. JSON UTF-8: `{"frame_id":1,"timestamp_ms":100}`.
3. Imagen JPEG o PNG comprimida.

Limite por mensaje: 6 MiB. Limite de imagen: 8.294.400 pixels.
Hay como maximo un cuadro pendiente de inferencia; un cuadro nuevo lo
reemplaza si la IA esta ocupada. Por eso no todos los IDs reciben respuesta.

Respuestas principales:

```json
{
  "type":"detections",
  "stream_id":"...",
  "frame_id":1,
  "timestamp_ms":100,
  "width":1280,
  "height":720,
  "detections":[{"class_name":"moto","confidence":0.91,"bbox":[100,200,300,450],"is_crash":false}],
  "crash_confirmed":false,
  "processing_ms":110,
  "dropped_frames":0
}
```

`helmet_detections` devuelve las detecciones de motos y cabezas obtenidas
en la segunda pasada. Puede llegar despues de `detections`; conserva el
`frame_id` y `timestamp_ms` originales. Las cajas de ambas respuestas son
`[x1,y1,x2,y2]` en pixels de la imagen **enviada**, no del recorte.
Si tu app escala el video para mostrarlo, escala tambien esas coordenadas.

Las alertas confirmadas se envian separadamente:

```json
{"type":"event","event":"no_casco","event_id":"...","stream_id":"...","frame_id":1,"timestamp_ms":100,"moto_id":1,"confidence":0.85,"bbox":[100,200,300,450],"snapshot_url":"/snapshots/no_casco_0123456789abcdef0123456789abcdef.jpg"}
```

Choques usan `event: crash`. Una deteccion `no_casco` aislada no es una
alerta: se mantiene la confirmacion por moto. No detectar casco no prueba
una infraccion; se exige detectar explicitamente `no_casco`.
Las capturas corresponden al cuadro que origino el evento.
Los errores de entrada usan `type: error`; la conexion sigue abierta.
Un fallo de procesamiento cierra la conexion con codigo 1011.
La reconexion inicia un seguimiento nuevo, sin mezclar motos de otra sesion.

`GET /events` conserva los ultimos 200 eventos **en memoria**; se borran
al reiniciar. Las capturas quedan en disco. Los cuadros y detecciones no
se guardan como video. La demora real depende de la PC y de la red.

## Cliente ejecutable de ejemplo

```powershell
.\.venv\Scripts\python.exe client_live.py --source 0
.\.venv\Scripts\python.exe client_live.py --source "C:\videos\ejemplo.mp4"
```

El cliente captura/envia cuadros mientras recibe resultados. No muestra
ventanas; imprime las respuestas para comprobar la integracion.
`encode_packet()` en `client_live.py` muestra exactamente como armar el
mensaje binario desde Python.

Ejemplo de envio desde JavaScript, con un Blob JPEG ya generado:

```javascript
const socket = new WebSocket('ws://127.0.0.1:8001/ws/live');
socket.onmessage = e => {
  const result = JSON.parse(e.data);
  // ready, detections, helmet_detections, event o error.
  console.log(result);
};
function sendFrame(jpegBlob, frameId, timestampMs) {
  if (socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 500000) return;
  const metadata = new TextEncoder().encode(JSON.stringify({
    frame_id: frameId, timestamp_ms: timestampMs
  }));
  const header = new ArrayBuffer(4);
  new DataView(header).setUint32(0, metadata.length, false);
  socket.send(new Blob([header, metadata, jpegBlob]));
}
```

## Configurar puerto o acceso desde otra PC

Por defecto escucha solo en esta PC. Copia `service_settings.example.json`
a `service_settings.json` para cambiar host, puerto o clave. Para otra PC,
usa `host: 0.0.0.0` y configura `api_key`; es obligatoria fuera de localhost.
Tu app conecta a la IP del equipo detector. El instalador no modifica el
firewall; la conectividad debe habilitarse segun la red donde lo despliegues.

HTTP: envia `x-api-key`. WebSocket: mismo header o `?token=CLAVE` para
clientes de navegador. Fuera de una red confiable, publica mediante un
proxy HTTPS/WSS. `/health` permite consultar estado sin clave.

Tambien se aceptan `DETECTION_HOST`, `DETECTION_PORT` y
`DETECTION_API_KEY`; tienen prioridad sobre el archivo de configuracion.
Para el servicio Windows es mas directo usar el archivo de configuracion.

Referencias: [WebSocket de FastAPI](https://fastapi.tiangolo.com/advanced/websockets/)
y [hosting de servicios pywin32](https://timgolden.me.uk/pywin32-docs/servicemanager.html).
