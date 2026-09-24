# OmniStage AI

Plataforma open source de **transcripción simultánea a escala** para conferencias. Nació para la **Nerdearla Vibeathon**: un emisor por escenario, una audiencia ilimitada por WebSocket, y **cero costo de APIs comerciales** porque el reconocimiento y la traducción corren en el edge.

OmniStage recibe audio del micrófono, descarta el silencio con un noise gate RMS (más un buffer circular de lookback para no cortar la primera sílaba), transcribe con **faster-whisper** y traduce **Español ↔ Inglés** con **Argos Translate**. El JSON `{original, translated}` se publica solo a quienes están mirando ese `stage_id`.

## Requisitos

- Python 3.10 o superior
- [ffmpeg](https://ffmpeg.org/) en el `PATH` (recomendado para decodificar los chunks WebM/Opus de `MediaRecorder`). Si no está, el emisor envía PCM 16 kHz como respaldo.
- Micrófono y un navegador moderno (Chrome, Edge o Firefox)

### ffmpeg en Windows

```powershell
winget install Gyan.FFmpeg
```

Cerrá y reabrí la terminal para que el `PATH` se actualice.

## Instalación local

```powershell
cd C:\Users\franc\Desktop\OmniStage_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```

La primera arrancada descarga el modelo Whisper (`tiny` por defecto, ~75 MB) y los pares de Argos `es↔en`. Después el nodo queda offline.

Abrí:

- Emisor: [http://127.0.0.1:8000/broadcaster](http://127.0.0.1:8000/broadcaster)
- Audiencia: [http://127.0.0.1:8000/audience](http://127.0.0.1:8000/audience)
- Salud del nodo: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

En otra máquina de la LAN usá la IP del host (`http://192.168.x.x:8000/...`). El WebSocket toma el host de la página.

### Variables de entorno

| Variable | Default | Rol |
|---|---|---|
| `OMNI_WHISPER_MODEL` | `tiny` | `tiny` o `base` (más calidad, más CPU) |
| `OMNI_WHISPER_DEVICE` | `cpu` | `cpu` o `cuda` |
| `OMNI_WHISPER_COMPUTE` | `int8` | `int8` en CPU, `float16` en GPU |
| `OMNI_RMS_THRESHOLD` | `0.015` | Umbral del noise gate |
| `OMNI_LOOKBACK_MS` | `280` | Lookback para no cortar la primera sílaba |
| `OMNI_PORT` | `8000` | Puerto HTTP/WS |

Ejemplo con el modelo `base`:

```powershell
$env:OMNI_WHISPER_MODEL="base"
python server.py
```

## Cómo se usa en una sala

1. El orador abre `/broadcaster`, escribe un `stage_id` (por ejemplo `nerdearla-main`) y pulsa **Iniciar micrófono**.
2. `MediaRecorder` corta el audio cada 1,5 s y lo manda por `/ws/broadcaster/{stage_id}`.
3. El servidor mide el RMS. Si el chunk está por debajo del umbral, se descarta y solo se conserva un lookback circular. Si hay voz, se antepone ese lookback y se manda a Whisper.
4. La audiencia abre `/audience`, elige el mismo escenario y el modo de lectura: **Original**, **Traducido** o **Ambos**.

### Browser Source en OBS Studio

En OBS: *Fuentes → Browser*, URL:

```
http://127.0.0.1:8000/audience?obs=1&stage=nerdearla-main&lang=both
```

El modo OBS usa fondo transparente y `text-shadow` para que los subtítulos se lean sobre la cámara o las diapositivas. También podés activarlo con el botón **Modo OBS** en la propia página.

## Arquitectura

```
Micrófono ──MediaRecorder──► /ws/broadcaster/{stage_id}
                                    │
                            Noise gate RMS
                            + lookback circular
                                    │
                         faster-whisper (tiny/base)
                                    │
                      Argos Translate (es ↔ en)
                                    │
                         /ws/audience/{stage_id} ──► navegadores / OBS
```

Un proceso FastAPI mantiene un mapa `stage_id → Stage`. Cada escenario tiene su cola `asyncio`, su worker de transcripción y su conjunto de sockets de audiencia. Los escenarios no se mezclan: el audio de `sala-2` nunca llega a quien está suscripto a `nerdearla-main`.

## Escalabilidad: varios escenarios, costo de API = 0

Esta solución escala **horizontalmente por escenario**, no por request a un proveedor de nube.

Whisper y Argos viven dentro del contenedor. No hay tokens de Google/OpenAI/DeepL, no hay facturación por minuto y no hay datos de voz saliendo del recinto. Eso es **Edge AI**: el modelo viaja hacia el evento, no el audio hacia una API.

Un diseño de producción para una conferencia con N salas:

1. **Un contenedor (o un pequeño pool) por `stage_id`.** Cada réplica carga `tiny` o `base` una sola vez y atiende a un emisor + N espectadores. El cuello de botella es CPU/RAM del nodo, no una cuota de API.
2. **Docker / Compose / Kubernetes** empaquetan `server.py`, el modelo y ffmpeg. En GPU se cambia `OMNI_WHISPER_DEVICE=cuda`.
3. **El balanceador enruta por path.** ` /ws/broadcaster/nerdearla-main` y `/ws/audience/nerdearla-main` deben caer siempre en el mismo proceso (sticky session por `stage_id`, o un servicio dedicado `omnistage-main`). Los WebSockets no se pueden repartir al azar entre workers sin un bus compartido.
4. **Si hace falta más de un worker en el mismo stage**, se publica el JSON en Redis Pub/Sub o NATS. El audio pesado sigue siendo local; solo viaja texto.
5. **La audiencia es barata.** Replicar un frame JSON a cientos de sockets es órdenes de magnitud más liviano que transcribir. Se puede separar el nodo de STT del nodo de fan-out.
6. **Auto-escala por salas activas**, no por espectadores: 8 escenarios = 8 contenedores. Apagar una sala libera el modelo.

En un Vibeathon esto cabe en una notebook. En un evento de varios escenarios, el mismo código se replica detrás de un reverse proxy. El costo marginal de un minuto extra de charla es electricidad, no una factura de API.

## Roadmap

### Fase 1 (esta entrega)

- Transcripción local con faster-whisper
- Traducción es ↔ en con Argos
- Noise gate RMS + lookback
- WebSockets multitenant
- Overlay OBS

### Fase 2

- **Gemma 2 (2B)** (modelos abiertos de Google DeepMind) sobre los textos ya transcritos para resúmenes automáticos por bloque, títulos de sala y un digest al cierre de cada charla. Gemma entra *después* de Whisper: no toca el audio y cabe en el mismo edge.
- **MediaPipe** para avatares de lengua de señas: a partir del texto (o de landmarks de un intérprete) generar una capa visual accesible, también en el navegador o en un sidecar, sin mandar video a un proveedor.

## 🤖 AI-Assisted Development

El código de este repositorio fue **co-creado con IA** (asistencia de un agente de programación para boilerplate, frontends y el cableado de FastAPI).

El diseño de la **arquitectura multitenant por `stage_id`**, el **noise gate con lookback circular** y la decisión de correr **Edge AI** (Whisper + Argos en el propio nodo para que el costo de API sea cero) son **trabajo humano**: son las restricciones de una conferencia real —varias salas, presupuesto cero de APIs, datos que no salen del recinto— las que definen el sistema.

## Licencia

[MIT](./LICENSE)
