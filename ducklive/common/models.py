"""Shared data models for DuckLive dashboard API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time


class ServiceStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class EngineStatus:
    """Status of a processing engine (face swap or voice change)."""

    enabled: bool = False
    loaded: bool = False
    model_name: str = ""
    avg_latency_ms: float = 0.0
    error: str | None = None


@dataclass
class StreamStats:
    """Real-time streaming statistics."""

    fps: float = 0.0
    bitrate_kbps: float = 0.0
    frame_count: int = 0
    dropped_frames: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class ClientInfo:
    """Connected client information."""

    id: str = ""
    address: str = ""
    connected_at: float = field(default_factory=time)
    frames_received: int = 0


@dataclass
class ServerState:
    """Global server state — the single source of truth for the dashboard."""

    status: ServiceStatus = ServiceStatus.STOPPED
    uptime_seconds: float = 0.0

    # Engines
    face_swap: EngineStatus = field(default_factory=EngineStatus)
    voice_change: EngineStatus = field(default_factory=EngineStatus)

    # Feed source
    feed_connected: bool = False
    feed_source: str = ""

    # Stream
    stream: StreamStats = field(default_factory=StreamStats)

    # Clients
    clients: list[ClientInfo] = field(default_factory=list)

    # GPU
    gpu_name: str = ""
    gpu_utilization_pct: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature_c: float = 0.0

    # Available assets
    available_faces: list[str] = field(default_factory=list)
    available_voices: list[str] = field(default_factory=list)
    current_face: str = ""
    current_voice: str = ""
