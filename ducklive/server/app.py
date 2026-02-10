"""DuckLive Server — main application orchestrator.

The server is a pure processing node:
  - Receives raw video/audio from feed clients (browser-based capture)
  - Processes with AI engines (face swap + voice change)
  - Broadcasts processed frames back to clients
  - Serves the dashboard for monitoring

In test mode, generates synthetic frames locally (no feed client needed).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ducklive.common.config import ServerConfig
from ducklive.common.models import EngineStatus, ServerState, ServiceStatus, StreamStats
from ducklive.common.protocol import DEFAULT_JPEG_QUALITY, Frame, compute_audio_level_db
from ducklive.server.discovery import ServiceAdvertiser
from ducklive.server.face_swap import FaceSwapEngine
from ducklive.server.head_swap import HeadSwapEngine
from ducklive.server.stream import StreamServer
from ducklive.server.voice_change import VoiceChangeEngine

logger = logging.getLogger(__name__)


class DuckLiveServer:
    """Main server application — receives raw frames, processes, streams."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.state = ServerState(status=ServiceStatus.STARTING)
        self._start_time = time.time()

        # Processing queues (feed clients push raw frames here)
        self.raw_video_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)
        self.raw_audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        # Components — HeadSwapEngine (LivePortrait) is default, FaceSwapEngine is fallback
        self.head_swap = HeadSwapEngine(weights_dir=None)  # auto-detect pretrained_weights/
        self.face_swap = FaceSwapEngine(model_dir=config.models_dir)  # fallback
        self.voice_change = VoiceChangeEngine(model_dir=config.models_dir)
        self.stream_server = StreamServer(
            host=config.host,
            port=config.ws_port,
            max_clients=config.max_clients,
            raw_video_queue=self.raw_video_queue,
            raw_audio_queue=self.raw_audio_queue,
        )
        self.advertiser = ServiceAdvertiser(
            ws_port=config.ws_port, dashboard_port=config.port
        )

        self._running = False
        self._video_task: asyncio.Task | None = None
        self._audio_task: asyncio.Task | None = None
        self._test_task: asyncio.Task | None = None

        # FPS tracking
        self._fps_timestamps: list[float] = []

    async def start(self) -> None:
        """Start all components."""
        logger.info("🦆 Starting DuckLive Server...")
        self._running = True

        # 1. Pre-load AI engines (no target set — client picks face/voice at runtime)
        # Try LivePortrait first (head swap), fall back to InsightFace inswapper
        try:
            logger.info("Loading LivePortrait head swap engine...")
            self.head_swap.load()
            logger.info("LivePortrait head swap engine ready (no target face set)")
        except Exception as e:
            logger.warning(f"LivePortrait unavailable: {e}, falling back to InsightFace inswapper")
            try:
                logger.info("Loading InsightFace face swap engine...")
                self.face_swap.load()
                logger.info("InsightFace face swap engine ready (no target face set)")
            except Exception as e2:
                logger.warning(f"Face swap engine unavailable: {e2}")

        try:
            logger.info("Loading voice change engine...")
            self.voice_change.load()
            logger.info("Voice change engine ready (no voice model set)")
        except Exception as e:
            logger.warning(f"Voice change engine unavailable: {e}")

        # 2. Start stream server (WebSocket)
        await self.stream_server.start()

        # 3. Advertise on network
        await self.advertiser.start()

        # 4. Start processing loops
        self._video_task = asyncio.create_task(self._process_video_loop())
        self._audio_task = asyncio.create_task(self._process_audio_loop())

        # 5. In test mode, generate synthetic frames
        if self.config.test_mode:
            self._test_task = asyncio.create_task(self._test_frame_generator())

        self.state.status = ServiceStatus.RUNNING
        logger.info("🦆 DuckLive Server is LIVE!")
        logger.info(f"   Dashboard:  http://0.0.0.0:{self.config.port}")
        logger.info(f"   WebSocket:  ws://0.0.0.0:{self.config.ws_port}")
        logger.info(f"   Feed:       ws://0.0.0.0:{self.config.ws_port}/feed")
        if self.config.test_mode:
            logger.info("   Mode:       TEST (synthetic frames)")
        else:
            logger.info("   Mode:       LIVE (waiting for feed client)")

    async def stop(self) -> None:
        """Stop all components gracefully."""
        logger.info("Stopping DuckLive Server...")
        self._running = False

        for task in (self._video_task, self._audio_task, self._test_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self.advertiser.stop()
        await self.stream_server.stop()

        self.state.status = ServiceStatus.STOPPED
        logger.info("DuckLive Server stopped.")

    def get_state(self) -> ServerState:
        """Get current server state for the dashboard."""
        self.state.uptime_seconds = time.time() - self._start_time

        # Face swap status — report whichever engine is active
        active_engine = self._active_face_engine
        self.state.face_swap = EngineStatus(
            enabled=self.config.face_swap_enabled,
            loaded=active_engine.is_loaded if active_engine else False,
            model_name=active_engine.model_name if active_engine else "",
            avg_latency_ms=active_engine.avg_latency_ms if active_engine else 0.0,
        )

        # Voice change status
        self.state.voice_change = EngineStatus(
            enabled=self.config.voice_change_enabled,
            loaded=self.voice_change.is_loaded,
            model_name=self.voice_change.model_name,
            avg_latency_ms=self.voice_change.avg_latency_ms,
        )

        # Feed source info
        self.state.feed_connected = self.stream_server.has_feed
        self.state.feed_source = self.stream_server.feed_source or ""
        if self.config.test_mode and not self.stream_server.has_feed:
            self.state.feed_source = "Test Pattern (synthetic)"

        # Stream stats
        self.state.stream = StreamStats(fps=self._compute_fps())
        self.state.clients = [
            type("ClientInfo", (), c)() for c in self.stream_server.clients_info
        ]

        # Available assets
        self.state.available_faces = self._list_assets(self.config.faces_dir, [".jpg", ".png"])
        self.state.available_voices = self._list_assets(self.config.voices_dir, [".pth"])
        self.state.current_face = (
            Path(active_engine.target_image_path).name if (active_engine and active_engine.has_target) else ""
        )
        self.state.current_voice = self.voice_change.model_name

        # GPU info
        self._update_gpu_info()

        return self.state

    async def set_face(self, image_path: str) -> bool:
        """Change the target face — uses whichever engine is active."""
        engine = self._active_face_engine
        if engine is None:
            return False
        return engine.set_target_face(image_path)

    async def set_voice(self, model_path: str) -> bool:
        """Change the voice model."""
        return self.voice_change.set_voice_model(model_path)

    @property
    def _active_face_engine(self) -> "HeadSwapEngine | FaceSwapEngine | None":
        """Return whichever face engine is loaded (LivePortrait preferred)."""
        if self.head_swap.is_loaded:
            return self.head_swap
        if self.face_swap.is_loaded:
            return self.face_swap
        return None

    # ─── Processing Loops ───

    async def _process_video_loop(self) -> None:
        """Process raw video frames from the queue."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]

        while self._running:
            try:
                # Block until a raw frame arrives
                jpeg_data = await asyncio.wait_for(
                    self.raw_video_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                # Decode JPEG → numpy array
                raw_frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
                )
                if raw_frame is None:
                    continue

                # Send original frame to dashboard
                await self.stream_server.broadcast_frame(
                    Frame.original_video(jpeg_data)
                )

                # Face swap (uses active engine — LivePortrait or InsightFace)
                # Run in thread pool to avoid blocking asyncio event loop
                # (LivePortrait on CPU can take seconds per frame)
                processed_frame = raw_frame
                active = self._active_face_engine
                if self.config.face_swap_enabled and active and active.is_loaded and active.has_target:
                    processed_frame = await asyncio.to_thread(
                        active.process_frame, raw_frame
                    )

                # Encode processed frame
                _, proc_jpeg = cv2.imencode(".jpg", processed_frame, encode_params)
                await self.stream_server.broadcast_frame(
                    Frame.video(proc_jpeg.tobytes())
                )

                # Track FPS
                now = time.monotonic()
                self._fps_timestamps.append(now)
                cutoff = now - 1.0
                self._fps_timestamps = [t for t in self._fps_timestamps if t > cutoff]

            except Exception as e:
                logger.error(f"Video processing error: {e}")

    async def _process_audio_loop(self) -> None:
        """Process raw audio chunks from the queue."""
        while self._running:
            try:
                raw_chunk = await asyncio.wait_for(
                    self.raw_audio_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                # Send original audio to dashboard
                await self.stream_server.broadcast_frame(
                    Frame.original_audio(raw_chunk)
                )

                # Voice change
                processed_chunk = raw_chunk
                if self.config.voice_change_enabled and self.voice_change.is_loaded:
                    result = self.voice_change.process_audio(raw_chunk)
                    if result:
                        processed_chunk = result

                # Send processed audio
                await self.stream_server.broadcast_frame(
                    Frame.audio(processed_chunk)
                )

                # Audio level meters
                levels = _json.dumps({
                    "original_db": round(compute_audio_level_db(raw_chunk), 1),
                    "processed_db": round(compute_audio_level_db(processed_chunk), 1),
                }).encode()
                await self.stream_server.broadcast_frame(Frame.audio_levels(levels))

            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    # ─── Test Mode ───

    async def _test_frame_generator(self) -> None:
        """Generate synthetic video + audio frames for testing (no feed client needed)."""
        from ducklive.common.protocol import AUDIO_CHUNK_SAMPLES, AUDIO_SAMPLE_RATE

        frame_interval = 1.0 / self.config.video_fps
        audio_chunk_interval = AUDIO_CHUNK_SAMPLES / AUDIO_SAMPLE_RATE
        frame_counter = 0
        audio_sample_idx = 0

        last_video = time.monotonic()
        last_audio = time.monotonic()

        logger.info("Test mode: generating synthetic frames")

        while self._running:
            now = time.monotonic()

            # Generate video frame at target FPS
            if now - last_video >= frame_interval:
                frame = self._generate_test_video(frame_counter)
                _, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
                try:
                    self.raw_video_queue.put_nowait(jpeg.tobytes())
                except asyncio.QueueFull:
                    pass  # Drop frame
                frame_counter += 1
                last_video = now

            # Generate audio chunk at target interval
            if now - last_audio >= audio_chunk_interval:
                audio = self._generate_test_audio(audio_sample_idx, AUDIO_CHUNK_SAMPLES)
                try:
                    self.raw_audio_queue.put_nowait(audio)
                except asyncio.QueueFull:
                    pass
                audio_sample_idx += AUDIO_CHUNK_SAMPLES
                last_audio = now

            await asyncio.sleep(0.001)

    def _generate_test_video(self, frame_num: int) -> np.ndarray:
        """Generate a synthetic test pattern frame."""
        w, h = self.config.video_width, self.config.video_height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark gradient background
        for y in range(h):
            frame[y, :] = [
                int(20 + 30 * y / h),
                int(15 + 25 * y / h),
                int(40 + 40 * y / h),
            ]

        t = frame_num

        # Moving circle ("face")
        cx = w // 2 + int(80 * np.sin(t * 0.05))
        cy = h // 2 + int(40 * np.cos(t * 0.03))
        cv2.circle(frame, (cx, cy), 80, (180, 200, 220), -1)
        cv2.circle(frame, (cx - 25, cy - 15), 10, (60, 60, 60), -1)
        cv2.circle(frame, (cx + 25, cy - 15), 10, (60, 60, 60), -1)
        cv2.ellipse(frame, (cx, cy + 25), (30, 15), 0, 0, 180, (60, 60, 60), 2)

        # Labels
        cv2.putText(frame, "DuckLive Test Pattern", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2)
        cv2.putText(frame, f"Frame #{t}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        return frame

    def _generate_test_audio(self, sample_idx: int, num_samples: int) -> bytes:
        """Generate a 440Hz sine wave test tone."""
        from ducklive.common.protocol import AUDIO_SAMPLE_RATE

        freq = 440.0
        t = np.arange(sample_idx, sample_idx + num_samples)
        samples = (3000 * np.sin(2 * np.pi * freq * t / AUDIO_SAMPLE_RATE)).astype(np.int16)
        return samples.tobytes()

    # ─── Helpers ───

    def _compute_fps(self) -> float:
        if len(self._fps_timestamps) < 2:
            return 0.0
        return float(len(self._fps_timestamps))

    def _list_assets(self, directory: Path, extensions: list[str]) -> list[str]:
        if not directory.exists():
            return []
        return sorted(f.name for f in directory.iterdir() if f.suffix.lower() in extensions)

    def _update_gpu_info(self) -> None:
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 5:
                    self.state.gpu_name = parts[0]
                    self.state.gpu_utilization_pct = float(parts[1])
                    self.state.gpu_memory_used_mb = float(parts[2])
                    self.state.gpu_memory_total_mb = float(parts[3])
                    self.state.gpu_temperature_c = float(parts[4])
        except Exception:
            pass


def start_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    test_mode: bool = False,
    dev: bool = False,
) -> None:
    """Entry point — start the DuckLive server."""
    logging.basicConfig(
        level=logging.DEBUG if dev else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ServerConfig(
        host=host,
        port=port,
        test_mode=test_mode,
    )

    server = DuckLiveServer(config)

    async def _run():
        await server.start()

        # Start dashboard (FastAPI)
        from ducklive.dashboard.api import create_app
        import uvicorn

        app = create_app(server)
        uvi_config = uvicorn.Config(
            app, host=host, port=port, log_level="debug" if dev else "info"
        )
        uvi_server = uvicorn.Server(uvi_config)

        # Register signal handlers for graceful shutdown (Unix only)
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

        try:
            await uvi_server.serve()
        finally:
            await server.stop()

    asyncio.run(_run())
