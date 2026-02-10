"""DuckLive Client — Web UI for browser-based camera capture + preview.

The browser:
  - Captures camera/mic via getUserMedia()
  - Sends raw frames to server via WebSocket (/feed)
  - Receives processed frames back for preview
  - Controls AI engines (select face/voice, toggle on/off)

The Python client provides:
  - /api/server-info → server WebSocket URL for the browser to connect
  - /api/status → client-side status (virtual devices, connection)
  - /api/faces, /api/voices, /api/engines → proxied from server
  - /api/faces/select, /api/voices/select, /api/engines/configure → proxied to server
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

if TYPE_CHECKING:
    from ducklive.client.app import DuckLiveClient

CLIENT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


def create_client_app(client: "DuckLiveClient") -> FastAPI:
    """Create the client Web UI."""

    app = FastAPI(title="DuckLive Client", version="0.3.0")

    static_dir = CLIENT_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(CLIENT_DIR / "templates"))

    # Shared HTTP client for proxying to server
    _http = httpx.AsyncClient(timeout=10.0)

    def _server_api(path: str) -> str:
        """Build full server API URL."""
        base = client.server_dashboard_url
        if not base:
            raise HTTPException(503, "Server not connected")
        return f"{base.rstrip('/')}{path}"

    # ─── Pages ───

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("client.html", {"request": request})

    # ─── API: Client Info ───

    @app.get("/api/server-info")
    async def server_info():
        """Provide the server WebSocket URL for the browser to connect directly."""
        ws_base = client.server_ws_base
        if ws_base:
            return {
                "feed_url": ws_base.rstrip("/") + "/feed",
                "stream_url": ws_base.rstrip("/") + "/stream",
                "server_address": _get_server_address(client),
                "dashboard_url": client.server_dashboard_url,
            }
        return {"feed_url": None, "stream_url": None, "server_address": None, "dashboard_url": None}

    @app.get("/api/status")
    async def get_status():
        """Client-side status — no server internals."""
        return {
            "server": {
                "url": client.server_ws_base or "not connected",
                "address": _get_server_address(client),
            },
            "receiver": {
                "connected": client.receiver.is_connected,
                "fps": round(client.receiver.fps, 1),
                "frames_received": client.receiver.frames_received,
            },
            "devices": {
                "virtual_camera": {
                    "enabled": client.virtual_cam is not None,
                    "running": client.virtual_cam.is_running if client.virtual_cam else False,
                    "name": client.virtual_cam.device_name if client.virtual_cam else "N/A",
                },
                "virtual_microphone": {
                    "enabled": client.virtual_mic is not None,
                    "running": client.virtual_mic.is_running if client.virtual_mic else False,
                },
            },
        }

    # ─── API: Proxy to Server — Assets ───

    @app.get("/api/faces")
    async def list_faces():
        """Proxy: list available face images from server."""
        try:
            r = await _http.get(_server_api("/api/faces"))
            return r.json()
        except Exception as e:
            logger.warning(f"Failed to fetch faces from server: {e}")
            return {"faces": [], "current": ""}

    @app.get("/api/faces/{name}/thumbnail")
    async def face_thumbnail(name: str):
        """Proxy: get face thumbnail from server."""
        try:
            r = await _http.get(_server_api(f"/api/faces/{name}/thumbnail"))
            if r.status_code == 200:
                return Response(content=r.content, media_type="image/jpeg")
            raise HTTPException(r.status_code, r.text)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Server error: {e}")

    @app.get("/api/voices")
    async def list_voices():
        """Proxy: list available voice models from server."""
        try:
            r = await _http.get(_server_api("/api/voices"))
            return r.json()
        except Exception as e:
            logger.warning(f"Failed to fetch voices from server: {e}")
            return {"voices": [], "current": ""}

    # ─── API: Proxy to Server — Selection ───

    @app.post("/api/faces/select")
    async def select_face(request: Request):
        """Proxy: select a face on the server."""
        body = await request.json()
        try:
            r = await _http.post(_server_api("/api/faces/select"), json=body)
            if r.status_code == 200:
                return r.json()
            raise HTTPException(r.status_code, r.json().get("detail", r.text))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Server error: {e}")

    @app.post("/api/voices/select")
    async def select_voice(request: Request):
        """Proxy: select a voice model on the server."""
        body = await request.json()
        try:
            r = await _http.post(_server_api("/api/voices/select"), json=body)
            if r.status_code == 200:
                return r.json()
            raise HTTPException(r.status_code, r.json().get("detail", r.text))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Server error: {e}")

    # ─── API: Proxy to Server — Engine Control ───

    @app.get("/api/engines")
    async def get_engines():
        """Proxy: get engine states from server."""
        try:
            r = await _http.get(_server_api("/api/engines"))
            return r.json()
        except Exception as e:
            logger.warning(f"Failed to fetch engine state: {e}")
            return {
                "face_swap": {"available": False, "enabled": False, "current_face": ""},
                "voice_change": {"available": False, "enabled": False, "current_voice": "", "pitch_shift": 0},
            }

    @app.post("/api/engines/configure")
    async def configure_engines(request: Request):
        """Proxy: configure engines on server."""
        body = await request.json()
        try:
            r = await _http.post(_server_api("/api/engines/configure"), json=body)
            if r.status_code == 200:
                return r.json()
            raise HTTPException(r.status_code, r.json().get("detail", r.text))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Server error: {e}")

    return app


def _get_server_address(client: "DuckLiveClient") -> str | None:
    if client.server_ws_base:
        parsed = urlparse(client.server_ws_base)
        return f"{parsed.hostname}:{parsed.port}" if parsed.hostname else None
    return None
