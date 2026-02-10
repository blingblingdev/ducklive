"""WebSocket stream receiver — receives video+audio from DuckLive server."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

import numpy as np

from ducklive.common.protocol import Frame, FrameType, HEADER_SIZE

logger = logging.getLogger(__name__)


class StreamReceiver:
    """Receives and decodes the DuckLive stream over WebSocket."""

    def __init__(self, max_video_buffer: int = 3, max_audio_buffer: int = 50):
        self._ws = None
        self._running = False

        # Buffers (thread-safe via asyncio)
        self._video_frames: deque[tuple[int, bytes]] = deque(maxlen=max_video_buffer)
        self._audio_chunks: deque[tuple[int, bytes]] = deque(maxlen=max_audio_buffer)

        # Stats
        self._frames_received = 0
        self._last_frame_time = 0.0
        self._fps_timestamps: deque[float] = deque(maxlen=60)

    async def connect(self, ws_url: str) -> None:
        """Connect to DuckLive server."""
        import websockets

        logger.info(f"Connecting to {ws_url}...")
        self._ws = await websockets.connect(
            ws_url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
        )
        self._running = True
        logger.info("Connected to DuckLive server!")

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def receive_loop(self) -> None:
        """Main receive loop — reads frames and dispatches to buffers."""
        while self._running and self._ws:
            try:
                data = await self._ws.recv()
                if not isinstance(data, bytes) or len(data) < HEADER_SIZE:
                    continue

                frame = Frame.unpack(data)
                now = time.monotonic()

                if frame.type == FrameType.VIDEO:
                    self._video_frames.append((frame.timestamp, frame.payload))
                    self._fps_timestamps.append(now)
                elif frame.type == FrameType.AUDIO:
                    self._audio_chunks.append((frame.timestamp, frame.payload))
                elif frame.type == FrameType.CONTROL:
                    logger.debug(f"Control message: {frame.payload.decode()}")

                self._frames_received += 1
                self._last_frame_time = now

            except Exception as e:
                if self._running:
                    logger.warning(f"Receive error: {e}")
                    break

    def get_video_frame(self) -> bytes | None:
        """Get the latest video frame (JPEG bytes). Non-blocking."""
        if self._video_frames:
            _, jpeg = self._video_frames[-1]
            self._video_frames.clear()  # Drop old frames
            return jpeg
        return None

    def get_audio_chunk(self) -> bytes | None:
        """Get the next audio chunk (PCM bytes). Non-blocking."""
        if self._audio_chunks:
            _, pcm = self._audio_chunks.popleft()
            return pcm
        return None

    def get_all_audio_chunks(self) -> list[bytes]:
        """Get all buffered audio chunks. Non-blocking."""
        chunks = [pcm for _, pcm in self._audio_chunks]
        self._audio_chunks.clear()
        return chunks

    @property
    def fps(self) -> float:
        if len(self._fps_timestamps) < 2:
            return 0.0
        elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return len(self._fps_timestamps) / elapsed

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    @property
    def frames_received(self) -> int:
        return self._frames_received
