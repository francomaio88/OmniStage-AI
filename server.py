"""
OmniStage AI — backend FastAPI para transcripción y traducción simultánea.

Arquitectura multitenant por stage_id:
  - /ws/broadcaster/{stage_id}  recibe chunks de audio (WebM/Opus o PCM s16le)
  - /ws/audience/{stage_id}     emite JSON {original, translated, ...}

El audio nunca viaja a una API comercial: Whisper y Helsinki-NLP (transformers) corren en el edge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ---------------------------------------------------------------------------
# Configuración (todas las constantes son ajustables por variables de entorno)
# ---------------------------------------------------------------------------

SAMPLE_RATE = int(os.getenv("OMNI_SAMPLE_RATE", "16000"))
RMS_THRESHOLD = float(os.getenv("OMNI_RMS_THRESHOLD", "0.05"))
LOOKBACK_MS = int(os.getenv("OMNI_LOOKBACK_MS", "280"))
LOOKBACK_SAMPLES = max(1, int(SAMPLE_RATE * LOOKBACK_MS / 1000))
HANGOVER_CHUNKS = int(os.getenv("OMNI_HANGOVER_CHUNKS", "3"))
WHISPER_MODEL = os.getenv("OMNI_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("OMNI_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("OMNI_WHISPER_COMPUTE", "int8")
QUEUE_MAX = int(os.getenv("OMNI_QUEUE_MAX", "6"))
HOST = os.getenv("OMNI_HOST", "0.0.0.0")
PORT = int(os.getenv("OMNI_PORT", "8000"))

WHISPER_HALLUCINATIONS = {
    "",
    ".",
    "..",
    "...",
    "thanks",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "you",
    "the",
    "gracias",
    "obrigado",
    "obrigada",
    "inscreva-se",
    "subtitles by",
    "subscribe",
}

logging.basicConfig(
    level=os.getenv("OMNI_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("omnistage")


# ---------------------------------------------------------------------------
# Buffer circular de lookback (evita cortar la primera sílaba)
# ---------------------------------------------------------------------------

class CircularLookback:
    """Buffer circular de muestras float32. Guarda los últimos N samples de silencio."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._filled = 0

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.size >= self.capacity:
            self._data[:] = samples[-self.capacity :]
            self._write = 0
            self._filled = self.capacity
            return
        n = samples.size
        end = self._write + n
        if end <= self.capacity:
            self._data[self._write : end] = samples
        else:
            first = self.capacity - self._write
            self._data[self._write :] = samples[:first]
            self._data[: n - first] = samples[first:]
        self._write = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def dump(self) -> np.ndarray:
        if self._filled == 0:
            return np.zeros(0, dtype=np.float32)
        if self._filled < self.capacity:
            return self._data[: self._filled].copy()
        return np.concatenate((self._data[self._write :], self._data[: self._write]))

    def clear(self) -> None:
        self._write = 0
        self._filled = 0


def rms_level(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


# ---------------------------------------------------------------------------
# Decodificación de audio (WebM/Opus vía ffmpeg, o PCM crudo)
# ---------------------------------------------------------------------------

FFMPEG_BIN = shutil.which("ffmpeg")
PCM_MAGIC = b"OMNIPCM1"


def _is_raw_pcm(payload: bytes) -> bool:
    return payload.startswith(PCM_MAGIC)


def decode_pcm_payload(payload: bytes) -> np.ndarray:
    """Payload propio: magic (8) + sample_rate u32le + int16le mono."""
    if len(payload) < 12:
        return np.zeros(0, dtype=np.float32)
    rate = struct.unpack_from("<I", payload, 8)[0]
    pcm = np.frombuffer(payload[12:], dtype="<i2").astype(np.float32) / 32768.0
    if rate and rate != SAMPLE_RATE:
        pcm = resample_linear(pcm, rate, SAMPLE_RATE)
    return pcm


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / src_rate
    target = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, target, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _ffmpeg_decode_sync(payload: bytes) -> np.ndarray:
    """Decodifica un chunk con ffmpeg en un hilo (compatible con el event loop de Windows)."""
    if not FFMPEG_BIN or not payload:
        return np.zeros(0, dtype=np.float32)
    proc = subprocess.run(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=payload,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        log.debug("ffmpeg rechazó un chunk: %s", err)
        return np.zeros(0, dtype=np.float32)
    if not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


async def decode_with_ffmpeg(payload: bytes) -> np.ndarray:
    return await asyncio.to_thread(_ffmpeg_decode_sync, payload)


async def decode_audio_chunk(payload: bytes) -> np.ndarray:
    if _is_raw_pcm(payload):
        return decode_pcm_payload(payload)
    return await decode_with_ffmpeg(payload)


# ---------------------------------------------------------------------------
# Whisper + Helsinki-NLP (se inicializan una sola vez al arrancar)
# ---------------------------------------------------------------------------

whisper_model = None
translator_pipelines: dict[tuple[str, str], dict] = {}
_translator_lock = threading.Lock()
# MODIFICADO: Eliminados los pares de portugués para el MVP
TRANSLATION_PAIRS = (("es", "en"), ("en", "es"))


def load_whisper():
    from faster_whisper import WhisperModel

    log.info(
        "Cargando faster-whisper model=%s device=%s compute=%s",
        WHISPER_MODEL,
        WHISPER_DEVICE,
        WHISPER_COMPUTE,
    )
    return WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )


def load_translators() -> dict[tuple[str, str], dict]:
    """Carga una sola vez los MarianMT de Helsinki-NLP y los deja en memoria."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = torch.device(
        "cuda" if torch.cuda.is_available() and WHISPER_DEVICE == "cuda" else "cpu"
    )
    cache: dict[tuple[str, str], dict] = {}
    for src, tgt in TRANSLATION_PAIRS:
        model_id = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        log.info("Cargando traductor local %s (device=%s)", model_id, device)
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id).to(device)
        model.eval()
        cache[(src, tgt)] = {"tokenizer": tokenizer, "model": model}
        log.info("Traductor %s listo", model_id)
    return cache


def translate_text(text: str, source: str, target: str) -> str:
    import torch

    text = text.strip()
    if not text or source == target:
        return text
    translator = translator_pipelines.get((source, target))
    if translator is None:
        log.warning("No hay modelo Helsinki-NLP para %s→%s", source, target)
        return text
    tokenizer = translator["tokenizer"]
    model = translator["model"]
    try:
        device = next(model.parameters()).device
        with _translator_lock:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
                translated_tokens = model.generate(**inputs)
                texto_traducido = tokenizer.decode(
                    translated_tokens[0], skip_special_tokens=True
                )
        return (texto_traducido or "").strip() or text
    except Exception as exc:
        log.warning("Fallo de traducción %s→%s: %s", source, target, exc)
        return text


# MODIFICADO: Solo permitimos es y en
ALLOWED_SPEAKER_LANGS = {"es", "en"}


def normalize_speaker_lang(value: Optional[str]) -> str:
    code = (value or "es").strip().lower().replace("_", "-")
    aliases = {
        "spanish": "es",
        "español": "es",
        "espanol": "es",
        "english": "en",
    }
    code = aliases.get(code, code.split("-", 1)[0])
    if code not in ALLOWED_SPEAKER_LANGS:
        return "es"
    return code


def transcribe_sync(audio: np.ndarray, language: str) -> tuple[str, str]:
    """Bloqueante: debe ejecutarse en un thread pool."""
    if whisper_model is None or audio.size < int(0.15 * SAMPLE_RATE):
        return "", language
    segments, info = whisper_model.transcribe(
        audio,
        language=language,
        beam_size=1,
        vad_filter=True,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.65,
        initial_prompt="A continuación se muestra una transcripción precisa del audio:"
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    lang = language or (info.language or "").lower()
    return text, lang


def pair_for_lang(detected: str) -> tuple[str, str]:
    # MODIFICADO: Eliminada la lógica de portugués
    if detected.startswith("es"):
        return "es", "en"
    if detected.startswith("en"):
        return "en", "es"
    return "es", "en"


def is_hallucination(text: str) -> bool:
    return text.strip().lower() in WHISPER_HALLUCINATIONS


# ---------------------------------------------------------------------------
# Estado por escenario (multitenant en proceso)
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    stage_id: str
    audience: set[WebSocket] = field(default_factory=set)
    broadcasters: set[WebSocket] = field(default_factory=set)
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_MAX))
    worker: Optional[asyncio.Task] = None
    lookback: CircularLookback = field(default_factory=lambda: CircularLookback(LOOKBACK_SAMPLES))
    hangover: int = 0
    speaking: bool = False
    speaker_lang: str = "es"
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


stages: dict[str, Stage] = {}
stages_lock = asyncio.Lock()


async def get_or_create_stage(stage_id: str) -> Stage:
    async with stages_lock:
        stage = stages.get(stage_id)
        if stage is None:
            stage = Stage(stage_id=stage_id)
            stages[stage_id] = stage
            stage.worker = asyncio.create_task(stage_worker(stage), name=f"worker-{stage_id}")
            log.info("Escenario creado: %s", stage_id)
        return stage


async def maybe_dispose_stage(stage_id: str) -> None:
    async with stages_lock:
        stage = stages.get(stage_id)
        if stage is None:
            return
        if stage.audience or stage.broadcasters:
            return
        stages.pop(stage_id, None)
        if stage.worker is not None:
            stage.worker.cancel()
        log.info("Escenario liberado: %s", stage_id)


async def enqueue_audio(stage: Stage, audio: np.ndarray) -> None:
    if audio.size == 0:
        return
    try:
        stage.queue.put_nowait(audio)
    except asyncio.QueueFull:
        try:
            _ = stage.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            stage.queue.put_nowait(audio)
        except asyncio.QueueFull:
            log.warning("Cola llena en %s; se descarta un chunk", stage.stage_id)


async def broadcast_stage(stage: Stage, payload: dict) -> None:
    message = json.dumps(payload, ensure_ascii=False)
    recipients = list(stage.audience | stage.broadcasters)
    stale: list[WebSocket] = []
    for ws in recipients:
        try:
            await ws.send_text(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        stage.audience.discard(ws)
        stage.broadcasters.discard(ws)


def apply_noise_gate(stage: Stage, audio: np.ndarray) -> Optional[np.ndarray]:
    """
    Umbral RMS + lookback circular + hangover.

    - Silencio: se descarta el chunk (ahorro de CPU) y se guarda en el buffer.
    - Ataque de voz: se antepone el lookback para no cortar la primera sílaba.
    - Caída de voz: se transcribe 1 chunk extra (hangover) para no cortar el final.
    """
    level = rms_level(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    voiced = level >= RMS_THRESHOLD
    decision = "PASA → Whisper" if voiced else "DESCARTADO (silencio)"
    msg = (
        f"[NOISE GATE] stage={stage.stage_id}  rms={level:.8f}  "
        f"peak={peak:.8f}  samples={audio.size}  umbral={RMS_THRESHOLD}  → {decision}"
    )
    print(msg, flush=True)
    log.info(msg)

    if voiced:
        prefix = stage.lookback.dump() if not stage.speaking else np.zeros(0, dtype=np.float32)
        stage.lookback.clear()
        stage.speaking = True
        stage.hangover = HANGOVER_CHUNKS
        if prefix.size:
            return np.concatenate((prefix, audio))
        return audio

    if stage.speaking and stage.hangover > 0:
        stage.hangover -= 1
        return audio

    stage.speaking = False
    stage.hangover = 0
    stage.lookback.push(audio)
    log.debug("Noise gate: silencio rms=%.4f stage=%s", level, stage.stage_id)
    return None


async def stage_worker(stage: Stage) -> None:
    loop = asyncio.get_running_loop()
    log.info("Worker de transcripción activo: %s", stage.stage_id)
    
    # AQUÍ ESTÁ LA MAGIA: Un buffer para guardar el audio mientras hablas
    audio_buffer = [] 
    
    try:
        while True:
            audio = await stage.queue.get()
            gated = apply_noise_gate(stage, audio)
            
            # Si el Noise Gate dice que estás hablando, guardamos el pedacito de audio
            if gated is not None:
                audio_buffer.append(gated)
                
            # ¿Cuándo traducimos? (AHORA SÍ ESTÁ DENTRO DEL WHILE)
            # 1. Si hubo silencio (gated es None) y hay algo grabado.
            # 2. O si el buffer alcanzó nuestro límite de latencia (3 chunks).
            if (gated is None and len(audio_buffer) > 0) or len(audio_buffer) >= 3:
                # Unimos todos los pedacitos en una sola grabación completa
                full_audio = np.concatenate(audio_buffer)
                audio_buffer.clear() # Vaciamos el buffer para tu próxima frase
                
                speaker_lang = stage.speaker_lang
                # Mandamos la frase completa a Whisper
                text, detected = await loop.run_in_executor(
                    None, transcribe_sync, full_audio, speaker_lang
                )
                text = (text or "").strip()
                if not text or is_hallucination(text):
                    continue
                
                # Mandamos la frase completa a traducir
                source, target = pair_for_lang(detected or speaker_lang)
                translated = await loop.run_in_executor(None, translate_text, text, source, target)
                translated = (translated or "").strip()
                
                payload = {
                    "stage_id": stage.stage_id,
                    "original": text,
                    "translated": translated,
                    "source_lang": source,
                    "target_lang": target,
                    "ts": time.time(),
                }
                log.info("[%s] %s: %s → %s", stage.stage_id, source, text, translated)
                await broadcast_stage(stage, payload)
                
    except asyncio.CancelledError:
        log.info("Worker detenido: %s", stage.stage_id)
        raise


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global whisper_model, translator_pipelines
    if FFMPEG_BIN:
        log.info("ffmpeg encontrado: %s", FFMPEG_BIN)
    else:
        log.warning(
            "ffmpeg no está en PATH. El emisor enviará PCM crudo como respaldo; "
            "instalar ffmpeg habilita la decodificación nativa de WebM/Opus."
        )
    loop = asyncio.get_running_loop()
    translator_pipelines = await loop.run_in_executor(None, load_translators)
    whisper_model = await loop.run_in_executor(None, load_whisper)
    log.info(
        "OmniStage listo | modelo=%s | rms_threshold=%.4f | lookback=%dms",
        WHISPER_MODEL,
        RMS_THRESHOLD,
        LOOKBACK_MS,
    )
    yield
    for stage in list(stages.values()):
        if stage.worker is not None:
            stage.worker.cancel()
    stages.clear()


app = FastAPI(
    title="OmniStage AI",
    description="Transcripción simultánea multitenant en el edge (Nerdearla Vibeathon).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>OmniStage AI</title>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; min-height:100vh; display:grid; place-items:center;
           font-family: "IBM Plex Sans", system-ui, sans-serif;
           background:#0e1116; color:#f4efe4; }
    main { text-align:center; padding:2rem; }
    h1 { font-weight:600; letter-spacing:.04em; margin:0 0 .4rem; }
    p { color:#b8b0a1; margin:0 0 1.6rem; }
    nav { display:flex; gap:.8rem; justify-content:center; flex-wrap:wrap; }
    a { color:#0e1116; background:#e8b86d; text-decoration:none;
        padding:.75rem 1.2rem; border-radius:999px; font-weight:600; }
    a.ghost { background:transparent; color:#e8b86d; border:1px solid #e8b86d; }
  </style>
</head>
<body>
  <main>
    <h1>OmniStage AI</h1>
    <p>Transcripción simultánea en el edge · Nerdearla Vibeathon</p>
    <nav>
      <a href="/broadcaster">Emisor</a>
      <a class="ghost" href="/audience">Espectador</a>
    </nav>
  </main>
</body>
</html>"""
    )


@app.get("/broadcaster")
async def broadcaster_page() -> FileResponse:
    return FileResponse(os.path.join(ROOT_DIR, "broadcaster.html"))


@app.get("/audience")
async def audience_page() -> FileResponse:
    return FileResponse(os.path.join(ROOT_DIR, "audience.html"))


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "whisper_model": WHISPER_MODEL,
            "whisper_loaded": whisper_model is not None,
            "ffmpeg": bool(FFMPEG_BIN),
            "stages": sorted(stages.keys()),
            "rms_threshold": RMS_THRESHOLD,
        }
    )


@app.get("/api/stages")
async def list_stages() -> JSONResponse:
    snapshot = []
    for sid, stage in stages.items():
        snapshot.append(
            {
                "stage_id": sid,
                "audience": len(stage.audience),
                "broadcasters": len(stage.broadcasters),
                "speaker_lang": stage.speaker_lang,
                "last_activity": stage.last_activity,
            }
        )
    return JSONResponse({"stages": snapshot})


@app.websocket("/ws/broadcaster/{stage_id}")
async def ws_broadcaster(websocket: WebSocket, stage_id: str, lang: str = "es") -> None:
    stage_id = stage_id.strip() or "default"
    speaker_lang = normalize_speaker_lang(lang or websocket.query_params.get("lang"))
    await websocket.accept()
    stage = await get_or_create_stage(stage_id)
    stage.speaker_lang = speaker_lang
    stage.broadcasters.add(websocket)
    stage.touch()
    log.info(
        "Broadcaster conectado a %s (lang=%s, %d emisores)",
        stage_id,
        speaker_lang,
        len(stage.broadcasters),
    )
    await websocket.send_text(
        json.dumps(
            {
                "type": "hello",
                "role": "broadcaster",
                "stage_id": stage_id,
                "speaker_lang": speaker_lang,
                "rms_threshold": RMS_THRESHOLD,
                "sample_rate": SAMPLE_RATE,
            }
        )
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            payload = message.get("bytes")
            if payload is None:
                continue
            stage.touch()
            audio = await decode_audio_chunk(payload)
            await enqueue_audio(stage, audio)
    except WebSocketDisconnect:
        pass
    finally:
        stage.broadcasters.discard(websocket)
        log.info("Broadcaster desconectado de %s", stage_id)
        await maybe_dispose_stage(stage_id)


@app.websocket("/ws/audience/{stage_id}")
async def ws_audience(websocket: WebSocket, stage_id: str) -> None:
    stage_id = stage_id.strip() or "default"
    await websocket.accept()
    stage = await get_or_create_stage(stage_id)
    stage.audience.add(websocket)
    stage.touch()
    log.info("Espectador conectado a %s (%d en sala)", stage_id, len(stage.audience))
    await websocket.send_text(
        json.dumps(
            {
                "type": "hello",
                "role": "audience",
                "stage_id": stage_id,
                "viewers": len(stage.audience),
            }
        )
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        stage.audience.discard(websocket)
        log.info("Espectador desconectado de %s", stage_id)
        await maybe_dispose_stage(stage_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)