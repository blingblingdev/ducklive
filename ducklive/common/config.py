"""DuckLive configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Server configuration (runs on GPU machine)."""

    model_config = {"env_prefix": "DUCKLIVE_"}

    # Network
    host: str = "0.0.0.0"
    port: int = 8080
    ws_port: int = 8765  # WebSocket stream port

    # Video processing
    video_width: int = 1280
    video_height: int = 720
    video_fps: int = 30
    jpeg_quality: int = 85

    # Audio processing
    audio_sample_rate: int = 16000

    # Face Swap
    face_swap_enabled: bool = False  # Off by default, client toggles on
    face_swap_model: str = "inswapper_128"  # InsightFace model name

    # Voice Change
    voice_change_enabled: bool = False  # Off by default, client toggles on
    voice_pitch_shift: int = 0  # semitones

    # Testing
    test_mode: bool = False  # Use synthetic video/audio instead of real devices

    # Performance
    max_clients: int = 5

    # Paths
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    faces_dir: Path = Field(default_factory=lambda: Path("faces"))
    voices_dir: Path = Field(default_factory=lambda: Path("voices"))


class ClientConfig(BaseSettings):
    """Client configuration (runs on Mac)."""

    model_config = {"env_prefix": "DUCKLIVE_CLIENT_"}

    # Server connection
    server_url: str | None = None  # None = auto-discover via mDNS
    server_timeout: float = 5.0

    # Virtual camera
    enable_camera: bool = True
    camera_name: str = "DuckLive Camera"
    camera_fps: int = 30

    # Virtual audio
    enable_audio: bool = True
    audio_device_name: str = "DuckLive Audio"

    # Client Web UI
    webui_port: int = 8081  # Client preview/status UI


# mDNS service config
MDNS_SERVICE_TYPE = "_ducklive._tcp.local."
MDNS_SERVICE_NAME = "DuckLive._ducklive._tcp.local."
