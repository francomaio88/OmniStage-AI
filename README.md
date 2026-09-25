# OmniStage AI

Plataforma **multitenant** de transcripción y traducción simultánea.

Creada para la **[Nerdearla Vibeathon](https://nerdear.la/)**: varias salas al mismo tiempo, un emisor por escenario, audiencia ilimitada por WebSocket, y un overlay listo para **OBS Studio**. El flujo principal de la demo es **Modo Nube (Gemini)**; el **Modo Local (BETA)** corre faster-whisper y Helsinki-NLP en CPU, sin costo de API.

Un orador habla al micrófono. OmniStage recorta el silencio, transcribe en el idioma del escenario y publica JSON `{original, translated, source_lang, target_lang}` solo a quienes están mirando ese `stage_id`.

---

## Características

- **Backend asíncrono** con **FastAPI** y **WebSockets** (`/ws/broadcaster/{stage_id}` y `/ws/audience/{stage_id}`). Cada escenario tiene su propia cola, su worker y su fan-out de sockets.
- **Modo Nube (Gemini)** por defecto en el emisor. Requiere `GEMINI_API_KEY`.
- **Modo Local (BETA)** opcional: **faster-whisper** (modelo **`base`**, `int8`) + **Helsinki-NLP** (`opus-mt-es-en` / `en-es`) en CPU.
- **VAD propio** (no solo el de Whisper): umbral RMS + **lookback circular** (~280 ms) para no cortar la primera sílaba + **hangover** de chunks para no cortar el final de la frase.
- **UI para OBS**: `audience.html` admite fondo transparente, tipografía con `text-shadow` y query `?obs=1`. `broadcaster.html` es la consola del emisor (micrófono, idioma, modo, eco de líneas).
- **Multitenant in-process**: el audio de `nerdearla-sala-2` nunca llega a quien está suscripto a `nerdearla-main`.
- **Modo Local (BETA)** opcional en el emisor, con etiqueta visual. El campo de contexto de Whisper solo se muestra en ese modo.

---

## Arquitectura

```
Micrófono ──MediaRecorder / PCM──► /ws/broadcaster/{stage_id}
                                         │
                              decode (ffmpeg o PCM)
                                         │
                         Noise gate RMS + lookback + hangover
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
             Local (edge, CPU)                         Nube (opcional)
      faster-whisper  modelo base                 Gemini (audio WAV)
      Helsinki-NLP  es ↔ en
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                              JSON {original, translated}
                                         │
                         /ws/audience/{stage_id} ──► navegador / OBS
```

Un proceso FastAPI mantiene `stage_id → Stage`. El audio pesado queda en el nodo; a la audiencia solo viaja texto.

---

## Requisitos

- **Python 3.11** (3.10+ también sirve)
- **FFmpeg en el PATH** — imprescindible para decodificar los micro-fragmentos WebM/Opus que manda el emisor. Si no está, el cliente cae a PCM 16 kHz como respaldo, pero el camino local “de verdad” usa FFmpeg.
- Micrófono y un navegador moderno (Chrome, Edge o Firefox)
- CPU con RAM suficiente para Whisper `base` + MarianMT (la primera corrida descarga los modelos)

---

## Instalación paso a paso (Windows)

### 1. FFmpeg

En PowerShell:

```powershell
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
```

Cerrá y reabrí la terminal (o Cursor) y comprobá:

```powershell
ffmpeg -version
```

### 2. Entorno virtual y dependencias

```powershell
git clone https://github.com/francomaio88/OmniStage-AI.git
cd OmniStage-AI

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` instala FastAPI, Uvicorn, faster-whisper, PyTorch, Transformers (Helsinki-NLP) y `google-genai` para el modo nube opcional.

### 3. Arrancar el servidor con Uvicorn

Con el venv **activado**:

```powershell
$env:GEMINI_API_KEY="tu_clave"
$env:OMNI_WHISPER_MODEL="base"
$env:OMNI_WHISPER_DEVICE="cpu"
$env:OMNI_WHISPER_COMPUTE="int8"

uvicorn server:app --host 0.0.0.0 --port 8000
```

Equivalente (usa el mismo `uvicorn.run` interno):

```powershell
$env:GEMINI_API_KEY="tu_clave"
$env:OMNI_WHISPER_MODEL="base"
python server.py
```

La primera vez descarga **faster-whisper `base`** y los Marian **es↔en**. Esperá a ver en el log que el traductor y Whisper están listos.

Abrí:

| Rol | URL |
|---|---|
| Home | http://127.0.0.1:8000/ |
| Emisor | http://127.0.0.1:8000/broadcaster |
| Audiencia | http://127.0.0.1:8000/audience |
| Salud | http://127.0.0.1:8000/health |

En otra máquina de la LAN usá la IP del host (`http://192.168.x.x:8000/...`). El WebSocket toma el host de la página.

---

## Cómo se usa en una sala

1. El orador abre `/broadcaster` (arranca en **Modo Nube (Gemini)**), elige idioma (Español o Inglés) y un `stage_id` (por ejemplo `nerdearla-main`).
2. Pulsa **Iniciar micrófono**. El navegador corta audio ~1 s y lo manda por WebSocket.
3. El servidor aplica el noise gate (umbral RMS muy bajo, apto para auriculares inalámbricos). Voz → Gemini (o Whisper `base` + Helsinki-NLP si elegís **Modo Local (BETA)**).
4. La audiencia abre `/audience`, elige el **mismo** escenario y el idioma de lectura: Original, Traducido o Ambos.

### Overlay en OBS Studio

*Fuentes → Browser*, URL:

```
http://127.0.0.1:8000/audience?obs=1&stage=nerdearla-main&lang=both
```

Fondo transparente, header y toolbar ocultos, subtítulos con sombra para leerse sobre cámara o slides. El botón **Modo OBS** en la propia página hace lo mismo.

---

## Variables de entorno

| Variable | Default en código | Rol |
|---|---|---|
| `OMNI_WHISPER_MODEL` | `small` | Usá **`base`** para la demo local en CPU |
| `OMNI_WHISPER_DEVICE` | `cpu` | `cpu` o `cuda` |
| `OMNI_WHISPER_COMPUTE` | `int8` | `int8` en CPU, `float16` en GPU |
| `OMNI_RMS_THRESHOLD` | `0.000005` | Umbral del noise gate (auriculares inalámbricos) |
| `OMNI_LOOKBACK_MS` | `280` | Lookback para no cortar el ataque de voz |
| `OMNI_HANGOVER_CHUNKS` | `3` | Chunks extra al caer la voz |
| `OMNI_HOST` / `OMNI_PORT` | `0.0.0.0` / `8000` | Bind de Uvicorn |
| `GEMINI_API_KEY` | vacío | **Requerida** para el Modo Nube (default) |
| `OMNI_GEMINI_MODEL` | `gemini-2.5-flash` | Modelo cloud opcional |

---

## Licencia

[MIT](./LICENSE)
