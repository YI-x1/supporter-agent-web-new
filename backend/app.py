from __future__ import annotations

import asyncio
import json
import os
import base64
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

import numpy as np

import os, sys

if sys.platform.startswith("win"):
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")


# Optional deps for streaming ASR
try:
    import webrtcvad
except Exception:
    webrtcvad = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


# ---------------------------
# SQLite (thread-safe minimal wrapper)
# ---------------------------

DB_PATH = os.environ.get("SUPPORTER_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "supporter.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_db_lock = threading.Lock()

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db() -> None:
    with _db_lock, _conn() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            ended_at INTEGER,
            summary_title TEXT,
            summary_text TEXT,
            final_result_json TEXT,
            meta_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            role TEXT NOT NULL,
            speaker TEXT,
            content TEXT NOT NULL,
            refined_content TEXT,
            ts INTEGER NOT NULL,
            extra_json TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );
        """)
        con.commit()

def db_create_session(meta: Optional[dict] = None) -> dict:
    sid = f"S_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    now = int(time.time())
    with _db_lock, _conn() as con:
        con.execute(
            "INSERT INTO sessions(session_id, created_at, meta_json) VALUES(?,?,?)",
            (sid, now, json.dumps(meta or {}, ensure_ascii=False)),
        )
        con.commit()
    return {"session_id": sid, "created_at": now}

def db_end_session(session_id: str, summary_title: Optional[str] = None,
                   summary_text: Optional[str] = None, final_result_json: Optional[dict] = None) -> None:
    now = int(time.time())
    with _db_lock, _conn() as con:
        con.execute(
            "UPDATE sessions SET ended_at=?, summary_title=COALESCE(?, summary_title), summary_text=COALESCE(?, summary_text), final_result_json=COALESCE(?, final_result_json) WHERE session_id=?",
            (now, summary_title, summary_text, json.dumps(final_result_json, ensure_ascii=False) if isinstance(final_result_json, dict) else final_result_json, session_id),
        )
        con.commit()

def db_list_sessions(limit: int = 50) -> List[dict]:
    with _db_lock, _conn() as con:
        rows = con.execute(
            "SELECT session_id, created_at, ended_at, summary_title FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    out = []
    for sid, c, e, title in rows:
        out.append({"session_id": sid, "created_at": c, "ended_at": e, "summary_title": title})
    return out

def db_add_message(session_id: str, phase: str, role: str, speaker: Optional[str],
                   content: str, refined: Optional[str] = None, extra: Optional[dict] = None) -> int:
    ts = int(time.time())
    with _db_lock, _conn() as con:
        cur = con.execute(
            "INSERT INTO messages(session_id, phase, role, speaker, content, refined_content, ts, extra_json) VALUES(?,?,?,?,?,?,?,?)",
            (session_id, phase, role, speaker, content, refined, ts, json.dumps(extra or {}, ensure_ascii=False))
        )
        con.commit()
        return int(cur.lastrowid)

def db_get_session(session_id: str) -> dict:
    with _db_lock, _conn() as con:
        row = con.execute(
            "SELECT session_id, created_at, ended_at, summary_title, summary_text, final_result_json, meta_json FROM sessions WHERE session_id=?",
            (session_id,)
        ).fetchone()
        if not row:
            raise KeyError("session_not_found")
        sid, created_at, ended_at, title, summary_text, final_json, meta_json = row

        msgs = con.execute(
            "SELECT id, phase, role, speaker, content, refined_content, ts, extra_json FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ).fetchall()

    messages = []
    for mid, phase, role, speaker, content, refined, ts, extra_json in msgs:
        messages.append({
            "id": mid, "phase": phase, "role": role, "speaker": speaker,
            "content": content, "refined_content": refined,
            "ts": ts, "extra": json.loads(extra_json) if extra_json else {}
        })

    return {
        "session_id": sid, "created_at": created_at, "ended_at": ended_at,
        "summary_title": title, "summary_text": summary_text,
        "final_result_json": json.loads(final_json) if final_json else None,
        "meta": json.loads(meta_json) if meta_json else {},
        "messages": messages
    }

def db_get_messages_after(session_id: str, after_id: int) -> List[dict]:
    with _db_lock, _conn() as con:
        msgs = con.execute(
            "SELECT id, phase, role, speaker, content, refined_content, ts, extra_json FROM messages WHERE session_id=? AND id>? ORDER BY id ASC",
            (session_id, after_id)
        ).fetchall()
    out = []
    for mid, phase, role, speaker, content, refined, ts, extra_json in msgs:
        out.append({
            "id": mid, "phase": phase, "role": role, "speaker": speaker,
            "content": content, "refined_content": refined,
            "ts": ts, "extra": json.loads(extra_json) if extra_json else {}
        })
    return out


# ---------------------------
# Real-time rooms (WebSocket hub)
# ---------------------------

@dataclass
class RoomState:
    masters: Set[WebSocket]
    clients: Set[WebSocket]
    names: Dict[str, str]   # channel -> name ("L"/"R")

rooms: Dict[str, RoomState] = {}

@dataclass
class AudioStreamState:
    vad: Any
    in_speech: bool
    speech_chunks: list[np.ndarray]
    prebuffer: list[np.ndarray]
    speech_run: int
    silence_run: int
    last_partial_ts: float
    last_voice_ts: float
    last_partial_text: str

    last_volume_ts: float
def make_audio_state() -> AudioStreamState:
    v = webrtcvad.Vad(2) if webrtcvad else None
    return AudioStreamState(
        vad=v, in_speech=False,
        speech_chunks=[],
        prebuffer=[],
        speech_run=0,
        silence_run=0,
        last_partial_ts=0.0,
        last_voice_ts=0.0,
        last_partial_text="",
        last_volume_ts=0.0
    )


rooms_lock = threading.Lock()

PING_INTERVAL_SEC = 20

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supporter-asr")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ASR gating to avoid short-audio hallucination
MIN_PARTIAL_SEC = float(os.environ.get("ASR_MIN_PARTIAL_SEC", "0.8"))
MIN_FINAL_SEC = float(os.environ.get("ASR_MIN_FINAL_SEC", "0.6"))
RMS_MIN = float(os.environ.get("ASR_RMS_MIN", "0.005"))
START_SPEECH_FRAMES = int(os.environ.get("ASR_START_SPEECH_FRAMES", "5"))
END_SILENCE_FRAMES = int(os.environ.get("ASR_END_SILENCE_FRAMES", "12"))



# Extra gating / cleanup to suppress tiny-noise hallucinations like "字幕by..."
MIN_TEXT_CHARS_FINAL = int(os.environ.get("ASR_MIN_TEXT_CHARS_FINAL", "2"))
MIN_TEXT_CHARS_PARTIAL = int(os.environ.get("ASR_MIN_TEXT_CHARS_PARTIAL", "1"))
RMS_MIN_FINAL = float(os.environ.get("ASR_RMS_MIN_FINAL", str(RMS_MIN)))
VOLUME_INTERVAL_SEC = float(os.environ.get("VOLUME_INTERVAL_SEC", "0.08"))  # ~12.5 FPS
VOLUME_GAIN = float(os.environ.get("VOLUME_GAIN", "8.0"))

NOISE_PATTERNS = [
    "字幕by", "字幕 by", "subtitles by", "字幕by索", "索兰娅", "solanya", "solania"
]

def _is_all_punct(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    return all(
        (not ch.isalnum())
        and (ch not in "一二三四五六七八九十零")
        and (not ("\u4e00" <= ch <= "\u9fff"))
        for ch in s
    )

def clean_and_filter_text(text: str, is_partial: bool = False) -> str:
    # Return cleaned text if usable, else '' to drop.
    if not text:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    low = t.lower().replace(" ", "")
    for p in NOISE_PATTERNS:
        if p.replace(" ", "").lower() in low:
            return ""
    if _is_all_punct(t):
        return ""
    min_len = MIN_TEXT_CHARS_PARTIAL if is_partial else MIN_TEXT_CHARS_FINAL
    if len(t) < min_len:
        return ""
    return t

# ---------------------------
# Streaming ASR (GPU)
# ---------------------------
ASR_MODEL_NAME = os.environ.get('ASR_MODEL', 'small')
ASR_DEVICE = os.environ.get('ASR_DEVICE', 'cuda')  # 'cuda' or 'cpu'
ASR_COMPUTE = os.environ.get('ASR_COMPUTE', 'float16')
ASR_LANGUAGE = os.environ.get('ASR_LANGUAGE', 'zh')
PARTIAL_INTERVAL_SEC = float(os.environ.get('ASR_PARTIAL_INTERVAL', '0.4'))
PARTIAL_WINDOW_SEC = float(os.environ.get('ASR_PARTIAL_WINDOW', '6'))
SILENCE_FINALIZE_SEC = float(os.environ.get('ASR_SILENCE_FINALIZE', '0.55'))
MAX_UTTERANCE_SEC = float(os.environ.get('ASR_MAX_UTTERANCE_SEC', '12'))

_whisper_model = None
_whisper_lock = threading.Lock()

def get_whisper_model():
    """Lazy-load ASR model.
    If CUDA init fails (common on Windows due to cuDNN), fallback to CPU so the pipeline still works.
    """
    global _whisper_model
    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model

        # First try configured device
        try:
            logger.info(f"Loading WhisperModel: model={ASR_MODEL_NAME}, device={ASR_DEVICE}, compute={ASR_COMPUTE}")
            _whisper_model = WhisperModel(ASR_MODEL_NAME, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
            return _whisper_model
        except Exception as e:
            logger.exception("Failed to load WhisperModel on configured device.")

            # Fallback: if cuda requested, try cpu int8 (slower but works without cuDNN)
            if str(ASR_DEVICE).lower().startswith("cuda"):
                try:
                    logger.warning("Falling back to CPU int8 (no cuDNN required). You can set ASR_DEVICE=cpu to silence this.")
                    _whisper_model = WhisperModel(ASR_MODEL_NAME, device="cpu", compute_type="int8")
                    return _whisper_model
                except Exception:
                    logger.exception("CPU fallback also failed.")
                    raise

            raise


def transcribe_text(audio_f32: np.ndarray) -> str:
    # audio_f32: float32 mono @16k
    if audio_f32 is None or audio_f32.size == 0:
        return ''
    model = get_whisper_model()
    segments, _info = model.transcribe(
        audio_f32,
        language=ASR_LANGUAGE,
        vad_filter=False,
        beam_size=3,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    text = ''.join([s.text for s in segments]).strip()
    return text


async def ws_send_safe(ws: WebSocket, obj: dict) -> None:
    try:
        await ws.send_text(json.dumps(obj, ensure_ascii=False))
    except Exception:
        # ignore; disconnect will be handled by receiver
        pass

async def broadcast_to_masters(session_id: str, obj: dict) -> None:
    with rooms_lock:
        room = rooms.get(session_id)
        targets = list(room.masters) if room else []
    for ws in targets:
        await ws_send_safe(ws, obj)

async def broadcast_to_clients(session_id: str, obj: dict) -> None:
    with rooms_lock:
        room = rooms.get(session_id)
        targets = list(room.clients) if room else []
    for ws in targets:
        await ws_send_safe(ws, obj)

def get_or_create_room(session_id: str) -> RoomState:
    with rooms_lock:
        if session_id not in rooms:
            rooms[session_id] = RoomState(masters=set(), clients=set(), names={"L": "说话人1", "R": "说话人2", "E": "老人"})
        return rooms[session_id]

def remove_ws(session_id: str, ws: WebSocket) -> None:
    with rooms_lock:
        room = rooms.get(session_id)
        if not room:
            return
        room.masters.discard(ws)
        room.clients.discard(ws)
        if not room.masters and not room.clients:
            rooms.pop(session_id, None)



def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    a = np.frombuffer(pcm, dtype=np.int16)
    return (a.astype(np.float32) / 32768.0)

def concat_audio(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(title="Supporter-Agent Web (Stable B)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()

class CreateSessionReq(BaseModel):
    meta: Optional[dict] = None

@app.post("/api/sessions")
def api_create_session(req: CreateSessionReq):
    s = db_create_session(req.meta or {})
    # ensure room exists (optional)
    get_or_create_room(s["session_id"])
    return s

@app.get("/api/sessions")
def api_list_sessions(limit: int = 50):
    return {"sessions": db_list_sessions(limit=limit)}

@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    try:
        return db_get_session(session_id)
    except KeyError:
        return JSONResponse({"error": "session_not_found"}, status_code=404)

@app.get("/api/sessions/{session_id}/messages")
def api_get_messages(session_id: str, after_id: int = 0):
    return {"messages": db_get_messages_after(session_id, after_id)}

class EndSessionReq(BaseModel):
    summary_title: Optional[str] = None
    summary_text: Optional[str] = None
    final_result_json: Optional[dict] = None

@app.post("/api/sessions/{session_id}/end")
def api_end_session(session_id: str, req: EndSessionReq):
    # Minimal "title" fallback: last user message snippet
    if req.summary_title is None:
        try:
            sess = db_get_session(session_id)
            last = ""
            for m in reversed(sess["messages"]):
                if m.get("speaker") and m.get("content"):
                    last = m["content"]
                    break
            req.summary_title = (last[:24] + "…") if len(last) > 24 else (last or "一次讨论")
        except Exception:
            req.summary_title = "一次讨论"
    db_end_session(session_id, req.summary_title, req.summary_text, req.final_result_json)
    return {"ok": True}

class AddMessageReq(BaseModel):
    phase: str = "discussion"
    role: str = "user"
    speaker: Optional[str] = None
    content: str
    refined_content: Optional[str] = None
    extra: Optional[dict] = None

@app.post("/api/sessions/{session_id}/messages")
def api_add_message(session_id: str, req: AddMessageReq):
    mid = db_add_message(session_id, req.phase, req.role, req.speaker, req.content, req.refined_content, req.extra)
    return {"id": mid}


# ---------------------------
# WebSocket endpoints
# ---------------------------

@app.websocket("/ws/master")
async def ws_master(ws: WebSocket, session_id: str = Query(...)):
    """Master (老人端) WebSocket.
    - Receives: set_active, text, names, pong
    - Sends: names, status, ping, text broadcasts, asr_partial (from clients)
    """
    await ws.accept()
    room = get_or_create_room(session_id)
    with rooms_lock:
        room.masters.add(ws)

    # notify master of current names and default active status
    await ws_send_safe(ws, {"type": "names", "L": room.names.get("L"), "R": room.names.get("R"), "E": room.names.get("E")})
    await ws_send_safe(ws, {"type": "status", "active": True, "session_id": session_id})

    last_ping = time.time()

    try:
        while True:
            # Heartbeat: server->client ping
            if time.time() - last_ping > PING_INTERVAL_SEC:
                await ws_send_safe(ws, {"type": "ping", "ts": int(time.time())})
                last_ping = time.time()

            try:
                event = await asyncio.wait_for(ws.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if event.get("text") is None:
                # Master side currently only sends text frames
                continue

            try:
                data = json.loads(event["text"])
            except Exception:
                continue

            if data.get("type") == "pong":
                continue

            # master can toggle client recording
            if data.get("type") == "set_active":
                active = bool(data.get("active"))
                await broadcast_to_clients(session_id, {"type": "status", "active": active})
                await broadcast_to_masters(session_id, {"type": "status", "active": active, "session_id": session_id})
                continue

            if data.get("type") == "names":
                with rooms_lock:
                    if "L" in data:
                        room.names["L"] = str(data["L"])
                    if "R" in data:
                        room.names["R"] = str(data["R"])
                    if "E" in data:
                        room.names["E"] = str(data["E"])
                    if "E" in data:
                        room.names["E"] = str(data["E"])
                await broadcast_to_masters(session_id, {"type": "names", "L": room.names.get("L"), "R": room.names.get("R"), "E": room.names.get("E")})
                await broadcast_to_clients(session_id, {"type": "names", "L": room.names.get("L"), "R": room.names.get("R"), "E": room.names.get("E")})
                continue

            # master sending text (elder)
            if data.get("type") == "text":
                speaker = data.get("speaker", "老人")
                content = data.get("content", "")
                phase = data.get("phase", "discussion")

                mid = db_add_message(session_id, phase, role="user", speaker=speaker, content=content, refined=None,
                                     extra={"from": "master"})
                await broadcast_to_masters(session_id, {
                    "type": "text", "id": mid, "phase": phase, "speaker": speaker, "content": content, "ts": int(time.time())
                })
                continue

    except WebSocketDisconnect:
        pass
    except Exception:
        # swallow to keep server stable; client will reconnect
        pass
    finally:
        remove_ws(session_id, ws)


@app.websocket("/ws/client")
async def ws_client(ws: WebSocket, session_id: str = Query(...), channel: str | None = Query(default=None)):
    """Client (采集端) WebSocket.
    - For 方案A(单设备双声道)：channel 参数可省略，一个网页同时上传 L/R。
    - For legacy: channel=L or R, and client can send JSON transcript.
    - Audio upload (binary):
        1 byte channel_id (0=L,1=R) + 4 bytes uint32 seq (LE) + PCM16 payload (16kHz mono, 10/20/30ms)
    """
    await ws.accept()
    room = get_or_create_room(session_id)

    channel = (channel or "DUAL").upper()
    if channel not in ("L", "R", "E", "DUAL"):
        channel = "DUAL"

    with rooms_lock:
        room.clients.add(ws)

    # send current status and names
    await ws_send_safe(ws, {"type": "status", "active": True})
    await ws_send_safe(ws, {"type": "names", "L": room.names.get("L"), "R": room.names.get("R"), "E": room.names.get("E")})

    last_ping = time.time()

    # per-channel audio streaming states (only used when receiving audio frames)
    audio_states = {"L": make_audio_state(), "R": make_audio_state(), "E": make_audio_state()}
    last_seq = {"L": 0, "R": 0, "E": 0}
    rx_frames = {"L": 0, "R": 0, "E": 0}
    asr_error_reported = {"L": False, "R": False, "E": False}

    try:
        while True:
            if time.time() - last_ping > PING_INTERVAL_SEC:
                await ws_send_safe(ws, {"type": "ping", "ts": int(time.time())})
                last_ping = time.time()

            try:
                event = await asyncio.wait_for(ws.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            data = None
            if event.get("text") is not None:
                try:
                    data = json.loads(event["text"])
                except Exception:
                    continue
            elif event.get("bytes") is not None:
                data = {"type": "audio_bytes", "bytes": event["bytes"]}
            else:
                continue

            if data.get("type") == "pong":
                continue

            # name update (from client page)
            if data.get("type") == "names":
                with rooms_lock:
                    if "L" in data:
                        room.names["L"] = str(data["L"])
                    if "R" in data:
                        room.names["R"] = str(data["R"])
                await broadcast_to_masters(session_id, {"type": "names", "L": room.names.get("L"), "R": room.names.get("R"), "E": room.names.get("E")})
                continue

            # --- audio streaming (PCM16 mono 16k) ---
            if data.get("type") in ("audio_bytes", "audio_chunk"):
                ch = "L"
                pcm = b""
                seq = 0

                if data.get("type") == "audio_bytes":
                    raw = data.get("bytes") or b""
                    if len(raw) < 5:
                        continue
                    ch_id = raw[0]
                    if ch_id == 0:
                        ch = "L"
                    elif ch_id == 1:
                        ch = "R"
                    elif ch_id == 2:
                        ch = "E"
                    else:
                        continue
                    seq = int.from_bytes(raw[1:5], "little", signed=False)
                    pcm = raw[5:]
                else:
                    ch = str(data.get("channel", "L")).upper()
                    if ch not in ("L", "R", "E"):
                        ch = "L"
                    try:
                        pcm = base64.b64decode(data.get("data_b64", ""))
                    except Exception:
                        continue

                last_seq[ch] = seq
                rx_frames[ch] += 1
                if rx_frames[ch] in (1, 50, 200):
                    logger.info(f"[ws_client] rx audio frame ch={ch} seq={seq} bytes={len(pcm)} frames={rx_frames[ch]}")
                st = audio_states[ch]
                now = time.time()

                frame_f32 = pcm16_bytes_to_float32(pcm)
                # --- volume broadcast (throttled) ---
                if frame_f32.size:
                    rms_frame = float(np.sqrt(np.mean(np.square(frame_f32))))
                else:
                    rms_frame = 0.0
                v = min(1.0, rms_frame * VOLUME_GAIN)
                if (now - st.last_volume_ts) >= VOLUME_INTERVAL_SEC:
                    st.last_volume_ts = now
                    await broadcast_to_masters(session_id, {"type": "volume", "channel": ch, "value": v})
                    await broadcast_to_clients(session_id, {"type": "volume", "channel": ch, "value": v})


                speech_frame = True
                if st.vad is not None:
                    try:
                        speech_frame = st.vad.is_speech(pcm, 16000)
                    except Exception:
                        speech_frame = True

                # Smooth VAD to avoid flapping / ultra-short utterances
                if speech_frame:
                    st.speech_run += 1
                    st.silence_run = 0
                else:
                    st.silence_run += 1
                    st.speech_run = 0

                # Not yet in speech: keep a small prebuffer; enter only after consecutive speech frames
                if not st.in_speech:
                    if speech_frame:
                        st.prebuffer.append(frame_f32)
                        if len(st.prebuffer) > START_SPEECH_FRAMES:
                            st.prebuffer.pop(0)
                        if st.speech_run >= START_SPEECH_FRAMES:
                            st.in_speech = True
                            st.speech_chunks = list(st.prebuffer)
                            st.prebuffer = []
                            st.last_partial_ts = 0.0
                            st.last_partial_text = ""
                            st.last_voice_ts = now
                    else:
                        if st.silence_run >= 2:
                            st.prebuffer = []
                    continue

                # In speech: append only speech frames
                if speech_frame:
                    st.last_voice_ts = now
                    st.speech_chunks.append(frame_f32)

                total_samples = sum(x.shape[0] for x in st.speech_chunks)

                # Partial transcription (throttled + min duration + energy gate + de-dup)
                if total_samples >= int(16000 * MIN_PARTIAL_SEC):
                    if st.last_partial_ts == 0.0 or (now - st.last_partial_ts) >= PARTIAL_INTERVAL_SEC:
                        st.last_partial_ts = now
                        audio = concat_audio(st.speech_chunks)
                        win = int(16000 * PARTIAL_WINDOW_SEC)
                        audio_win = audio[-win:] if audio.shape[0] > win else audio
                        rms = float(np.sqrt(np.mean(np.square(audio_win)))) if audio_win.size else 0.0
                        if rms >= RMS_MIN:
                            try:
                                text = transcribe_text(audio_win)
                            except Exception:
                                if not asr_error_reported[ch]:
                                    logger.exception(f"ASR partial failed for ch={ch}. Check CUDA/cuDNN/cublas or use ASR_DEVICE=cpu.")
                                    asr_error_reported[ch] = True
                                text = ""
                            text = clean_and_filter_text((text or "").strip(), is_partial=True)
                            if text and text != st.last_partial_text:
                                st.last_partial_text = text
                                speaker = room.names.get(ch) or ("说话人1" if ch == "L" else ("说话人2" if ch == "R" else "老人"))
                                await broadcast_to_masters(session_id, {
                                    "type": "asr_partial",
                                    "phase": "discussion",
                                    "speaker": speaker,
                                    "channel": ch,
                                    "content": text,
                                    "ts": int(time.time())
                                })

                # Finalize: end by silence
                if st.silence_run >= END_SILENCE_FRAMES:
                    if total_samples >= int(16000 * MIN_FINAL_SEC):
                        audio = concat_audio(st.speech_chunks)
                        try:
                            final_text = transcribe_text(audio)
                        except Exception:
                            if not asr_error_reported[ch]:
                                logger.exception(f"ASR final failed for ch={ch}. Check CUDA/cuDNN/cublas or use ASR_DEVICE=cpu.")
                                asr_error_reported[ch] = True
                            final_text = ""
                        final_text = clean_and_filter_text((final_text or "").strip(), is_partial=False)

                        rms_final = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
                        if rms_final < RMS_MIN_FINAL:
                            final_text = ""

                        if final_text:
                            speaker = room.names.get(ch) or ("说话人1" if ch == "L" else ("说话人2" if ch == "R" else "老人"))
                            mid = db_add_message(
                                session_id, "discussion", role="user", speaker=speaker,
                                content=final_text, refined=None,
                                extra={"from": ("elder_mic" if ch=="E" else "client_dual"), "channel": ch, "seq": last_seq.get(ch, 0)}
                            )
                            await broadcast_to_masters(session_id, {
                                "type": "text", "id": mid, "phase": "discussion",
                                "speaker": speaker, "content": final_text, "ts": int(time.time())
                            })

                    # reset even if too short (drop)
                    st.in_speech = False
                    st.speech_chunks = []
                    st.prebuffer = []
                    st.speech_run = 0
                    st.silence_run = 0
                    st.last_partial_ts = 0.0
                    st.last_partial_text = ""
                    continue

                # Otherwise keep accumulating
                continue

# Backward compatible: old client sends {speaker, content} or {type:"transcript"...}
            if data.get("type") in (None, "", "transcript"):
                speaker = (
                    data.get("speaker")
                    or (room.names.get(channel) if channel in ("L", "R") else None)
                    or ("说话人" + (channel if channel != "DUAL" else ""))
                )
                content = data.get("content", "")
                phase = data.get("phase", "discussion")

                mid = db_add_message(
                    session_id, phase, role="user", speaker=speaker, content=content,
                    refined=None, extra={"from": ("client_dual" if channel == "DUAL" else f"client_{channel}")}
                )

                await broadcast_to_masters(session_id, {
                    "type": "text", "id": mid, "phase": phase,
                    "speaker": speaker, "content": content, "ts": int(time.time())
                })
                continue

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        remove_ws(session_id, ws)


# Serve static pages (master/client)
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "public")
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
