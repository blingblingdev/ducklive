"""DuckLive Dashboard — FastAPI backend.

Serves two roles:
  1. Dashboard Web UI for server monitoring
  2. REST API for clients to query available assets and control engines
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

if TYPE_CHECKING:
    from ducklive.server.app import DuckLiveServer

DASHBOARD_DIR = Path(__file__).parent
THUMB_SIZE = (128, 128)
FACE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VOICE_EXTENSIONS = {".pth"}


def create_app(server: "DuckLiveServer") -> FastAPI:
    """Create the FastAPI dashboard + API application."""

    app = FastAPI(title="DuckLive Server", version="0.2.0")

    # Mount static files
    static_dir = DASHBOARD_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(DASHBOARD_DIR / "templates"))

    # ─── Dashboard Page ───

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

    # ─── API: Server State (monitoring) ───

    @app.get("/api/state")
    async def get_state():
        """Full server state for dashboard polling."""
        state = server.get_state()
        return _serialize_state(state)

    # ─── API: Available Assets ───

    @app.get("/api/faces")
    async def list_faces():
        """List available face images in the server's faces/ directory."""
        faces = _list_files(server.config.faces_dir, FACE_EXTENSIONS)
        engine = server._active_face_engine
        current = engine.target_image_path if (engine and engine.has_target) else ""
        return {
            "faces": faces,
            "current": Path(current).name if current else "",
        }

    @app.get("/api/faces/{name}/thumbnail")
    async def face_thumbnail(name: str):
        """Return a small JPEG thumbnail of a face image."""
        face_path = server.config.faces_dir / name
        if not face_path.exists() or face_path.suffix.lower() not in FACE_EXTENSIONS:
            raise HTTPException(404, "Face not found")

        img = cv2.imread(str(face_path))
        if img is None:
            raise HTTPException(422, "Cannot read image")

        # Resize to thumbnail
        h, w = img.shape[:2]
        scale = min(THUMB_SIZE[0] / w, THUMB_SIZE[1] / h)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.get("/api/voices")
    async def list_voices():
        """List available voice models in the server's voices/ directory."""
        voices = _list_files(server.config.voices_dir, VOICE_EXTENSIONS)
        current = server.voice_change.model_name
        return {
            "voices": voices,
            "current": current,
        }

    # ─── API: Asset Selection (called by client) ───

    @app.post("/api/faces/select")
    async def select_face(request: Request):
        """Select a face image as the swap target."""
        body = await request.json()
        filename = body.get("filename", "").strip()

        # Empty filename = clear target
        if not filename:
            # Clear on whichever engine is active
            if server.head_swap.is_loaded:
                server.head_swap._source_feature_3d = None
                server.head_swap._target_image_path = ""
                server.head_swap.reset_driving()
            if server.face_swap.is_loaded:
                server.face_swap._target_face = None
                server.face_swap._target_image_path = ""
            return {"status": "ok", "face": ""}

        face_path = server.config.faces_dir / filename
        if not face_path.exists():
            raise HTTPException(404, f"Face not found: {filename}")

        engine = server._active_face_engine
        if engine is None or not engine.is_loaded:
            raise HTTPException(503, "Face swap engine not loaded")

        success = await server.set_face(str(face_path))
        if not success:
            raise HTTPException(422, "No face detected in image")

        return {"status": "ok", "face": filename}

    @app.post("/api/voices/select")
    async def select_voice(request: Request):
        """Select a voice model."""
        body = await request.json()
        filename = body.get("filename", "").strip()

        # Empty filename = clear model
        if not filename:
            server.voice_change._model_path = ""
            return {"status": "ok", "voice": ""}

        voice_path = server.config.voices_dir / filename
        if not voice_path.exists():
            raise HTTPException(404, f"Voice model not found: {filename}")

        if not server.voice_change.is_loaded:
            raise HTTPException(503, "Voice change engine not loaded")

        success = await server.set_voice(str(voice_path))
        if not success:
            raise HTTPException(422, "Failed to load voice model")

        return {"status": "ok", "voice": filename}

    # ─── API: Engine Configuration (called by client) ───

    @app.get("/api/engines")
    async def get_engines():
        """Get current engine states — what the client needs to render its control panel."""
        engine = server._active_face_engine
        return {
            "face_swap": {
                "available": engine.is_loaded if engine else False,
                "enabled": server.config.face_swap_enabled,
                "engine_type": engine.model_name if engine else "",
                "current_face": Path(engine.target_image_path).name if (engine and engine.has_target) else "",
            },
            "voice_change": {
                "available": server.voice_change.is_loaded,
                "enabled": server.config.voice_change_enabled,
                "current_voice": server.voice_change.model_name,
                "pitch_shift": server.config.voice_pitch_shift,
            },
        }

    @app.post("/api/engines/configure")
    async def configure_engines(request: Request):
        """Update engine configuration from the client.

        Accepts any subset of:
          face_swap_enabled: bool
          voice_change_enabled: bool
          voice_pitch_shift: int (-12..+12)
        """
        body = await request.json()

        if "face_swap_enabled" in body:
            server.config.face_swap_enabled = bool(body["face_swap_enabled"])

        if "voice_change_enabled" in body:
            server.config.voice_change_enabled = bool(body["voice_change_enabled"])

        if "voice_pitch_shift" in body:
            server.config.voice_pitch_shift = max(-12, min(12, int(body["voice_pitch_shift"])))
            server.voice_change.pitch_shift = server.config.voice_pitch_shift

        return {
            "status": "ok",
            "face_swap_enabled": server.config.face_swap_enabled,
            "voice_change_enabled": server.config.voice_change_enabled,
            "voice_pitch_shift": server.config.voice_pitch_shift,
        }

    # ─── API: Preview WebSocket URL ───

    @app.get("/api/preview/url")
    async def preview_url():
        return {"ws_url": f"ws://{{host}}:{server.config.ws_port}/dashboard"}

    return app


# ─── Helpers ───


def _list_files(directory: Path, extensions: set[str]) -> list[str]:
    """List files in a directory matching the given extensions."""
    if not directory.exists():
        return []
    return sorted(
        f.name for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )


def _serialize_state(state) -> dict:
    """Convert ServerState dataclass to JSON-serializable dict."""
    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    try:
        return _to_dict(state)
    except Exception:
        return {
            "status": state.status.value if hasattr(state.status, "value") else str(state.status),
            "uptime_seconds": state.uptime_seconds,
        }
