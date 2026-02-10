"""DuckLive streaming protocol — binary frame format over WebSocket.

Frame format:
    ┌──────┬───────────┬──────────┬─────────────┐
    │ Type │ Timestamp │ Size     │ Payload     │
    │ 1B   │ 8B (u64)  │ 4B (u32) │ variable    │
    └──────┴───────────┴──────────┴─────────────┘

Downstream (server → client):
    0x01 = Processed video frame (JPEG) — sent to feed + stream clients
    0x02 = Processed audio chunk (PCM)  — sent to feed + stream clients
    0x03 = Control message (JSON)
    0x04 = Original video frame (JPEG)  — dashboard only
    0x05 = Original audio chunk (PCM)   — dashboard only
    0x06 = Audio level meter (JSON)     — dashboard only

Upstream (client → server):
    0x10 = Raw video frame (JPEG)       — feed client sends to server
    0x11 = Raw audio chunk (PCM s16le)  — feed client sends to server
"""

import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Self


class FrameType(IntEnum):
    # Downstream: server → client
    VIDEO = 0x01            # Processed video (after face swap)
    AUDIO = 0x02            # Processed audio (after voice change)
    CONTROL = 0x03          # Control message (JSON)
    ORIGINAL_VIDEO = 0x04   # Original video (before face swap) — dashboard only
    ORIGINAL_AUDIO = 0x05   # Original audio (before voice change) — dashboard only
    AUDIO_LEVELS = 0x06     # Audio level meters (JSON: {original_db, processed_db})
    # Upstream: client → server
    RAW_VIDEO = 0x10        # Raw video frame from client camera (JPEG)
    RAW_AUDIO = 0x11        # Raw audio chunk from client microphone (PCM s16le)


# Header: type (1B) + timestamp (8B) + size (4B) = 13 bytes
HEADER_FORMAT = "!BQI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Frame types that are only sent to dashboard, not to regular clients
DASHBOARD_ONLY_TYPES = {FrameType.ORIGINAL_VIDEO, FrameType.ORIGINAL_AUDIO, FrameType.AUDIO_LEVELS}

# Frame types sent upstream from feed clients
UPSTREAM_TYPES = {FrameType.RAW_VIDEO, FrameType.RAW_AUDIO}


@dataclass(slots=True)
class Frame:
    type: FrameType
    timestamp: int  # microseconds since epoch
    payload: bytes

    def pack(self) -> bytes:
        """Serialize frame to binary."""
        header = struct.pack(HEADER_FORMAT, self.type, self.timestamp, len(self.payload))
        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> Self:
        """Deserialize frame from binary."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Frame too short: {len(data)} < {HEADER_SIZE}")
        frame_type, timestamp, size = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        payload = data[HEADER_SIZE : HEADER_SIZE + size]
        if len(payload) != size:
            raise ValueError(f"Payload size mismatch: {len(payload)} != {size}")
        return cls(type=FrameType(frame_type), timestamp=timestamp, payload=payload)

    @classmethod
    def video(cls, jpeg_data: bytes) -> Self:
        """Create a processed video frame."""
        return cls(type=FrameType.VIDEO, timestamp=_now_us(), payload=jpeg_data)

    @classmethod
    def audio(cls, pcm_data: bytes) -> Self:
        """Create a processed audio frame."""
        return cls(type=FrameType.AUDIO, timestamp=_now_us(), payload=pcm_data)

    @classmethod
    def control(cls, json_bytes: bytes) -> Self:
        """Create a control frame."""
        return cls(type=FrameType.CONTROL, timestamp=_now_us(), payload=json_bytes)

    @classmethod
    def original_video(cls, jpeg_data: bytes) -> Self:
        """Create an original (pre-processing) video frame."""
        return cls(type=FrameType.ORIGINAL_VIDEO, timestamp=_now_us(), payload=jpeg_data)

    @classmethod
    def original_audio(cls, pcm_data: bytes) -> Self:
        """Create an original (pre-processing) audio frame."""
        return cls(type=FrameType.ORIGINAL_AUDIO, timestamp=_now_us(), payload=pcm_data)

    @classmethod
    def audio_levels(cls, json_bytes: bytes) -> Self:
        """Create an audio level meter frame."""
        return cls(type=FrameType.AUDIO_LEVELS, timestamp=_now_us(), payload=json_bytes)

    @classmethod
    def raw_video(cls, jpeg_data: bytes) -> Self:
        """Create a raw video frame (from client camera)."""
        return cls(type=FrameType.RAW_VIDEO, timestamp=_now_us(), payload=jpeg_data)

    @classmethod
    def raw_audio(cls, pcm_data: bytes) -> Self:
        """Create a raw audio frame (from client microphone)."""
        return cls(type=FrameType.RAW_AUDIO, timestamp=_now_us(), payload=pcm_data)


def _now_us() -> int:
    """Current time in microseconds."""
    return int(time.time() * 1_000_000)


def compute_audio_level_db(pcm_data: bytes) -> float:
    """Compute RMS audio level in dB from PCM s16le data."""
    import numpy as np

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1.0:
        return -100.0
    # dBFS (relative to max int16 = 32768)
    return float(20 * np.log10(rms / 32768.0))


# Audio config constants
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2  # s16le = 2 bytes per sample
AUDIO_CHUNK_MS = 20  # 20ms chunks for low latency
AUDIO_CHUNK_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS // 1000  # 320 samples
AUDIO_CHUNK_BYTES = AUDIO_CHUNK_SAMPLES * AUDIO_SAMPLE_WIDTH * AUDIO_CHANNELS  # 640 bytes

# Video config defaults
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_FPS = 30
DEFAULT_JPEG_QUALITY = 85
