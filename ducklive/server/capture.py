"""Video and audio capture from local devices."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ducklive.common.protocol import (
    AUDIO_CHUNK_SAMPLES,
    AUDIO_SAMPLE_RATE,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
)


@dataclass
class CaptureConfig:
    camera_index: int = 0
    width: int = DEFAULT_VIDEO_WIDTH
    height: int = DEFAULT_VIDEO_HEIGHT
    fps: int = DEFAULT_VIDEO_FPS
    audio_device_index: int | None = None
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    test_mode: bool = False  # Use synthetic test pattern instead of real camera/mic


class VideoCapture:
    """Threaded video capture from webcam (or synthetic test pattern)."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps_counter = _FPSCounter()
        self._test_mode = config.test_mode
        self._frame_counter = 0

    def start(self) -> None:
        if self._running:
            return

        if not self._test_mode:
            self._cap = cv2.VideoCapture(self.config.camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.config.camera_index}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self) -> np.ndarray | None:
        """Get the latest frame (non-blocking)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    @property
    def resolution(self) -> str:
        if self._test_mode:
            return f"{self.config.width}x{self.config.height}"
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return f"{w}x{h}"
        return "N/A"

    @property
    def camera_name(self) -> str:
        if self._test_mode:
            return "Test Pattern (synthetic)"
        return f"Camera #{self.config.camera_index}"

    def _capture_loop(self) -> None:
        target_interval = 1.0 / self.config.fps
        while self._running:
            t0 = time.monotonic()

            if self._test_mode:
                frame = self._generate_test_frame()
            else:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue

            with self._lock:
                self._frame = frame
            self._fps_counter.tick()

            if self._test_mode:
                elapsed = time.monotonic() - t0
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _generate_test_frame(self) -> np.ndarray:
        """Generate a synthetic test pattern with moving elements."""
        w, h = self.config.width, self.config.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark gradient background
        for y in range(h):
            frame[y, :] = [int(20 + 30 * y / h), int(15 + 25 * y / h), int(40 + 40 * y / h)]

        self._frame_counter += 1
        t = self._frame_counter

        # Moving circle (simulates a "face")
        cx = w // 2 + int(80 * np.sin(t * 0.05))
        cy = h // 2 + int(40 * np.cos(t * 0.03))
        cv2.circle(frame, (cx, cy), 80, (180, 200, 220), -1)
        # Eyes
        cv2.circle(frame, (cx - 25, cy - 15), 10, (60, 60, 60), -1)
        cv2.circle(frame, (cx + 25, cy - 15), 10, (60, 60, 60), -1)
        # Mouth
        cv2.ellipse(frame, (cx, cy + 25), (30, 15), 0, 0, 180, (60, 60, 60), 2)

        # Label
        cv2.putText(frame, "DuckLive Test Pattern", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2)
        cv2.putText(frame, f"Frame #{t}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        return frame


class AudioCapture:
    """Threaded audio capture from microphone (or synthetic test tone)."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._stream = None
        self._pa = None
        self._buffer: list[bytes] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._test_mode = config.test_mode
        self._test_sample_idx = 0

    def start(self) -> None:
        if self._running:
            return

        if not self._test_mode:
            import pyaudio

            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.audio_sample_rate,
                input=True,
                input_device_index=self.config.audio_device_index,
                frames_per_buffer=AUDIO_CHUNK_SAMPLES,
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    def read_chunks(self) -> list[bytes]:
        """Get all buffered audio chunks and clear the buffer."""
        with self._lock:
            chunks = self._buffer[:]
            self._buffer.clear()
        return chunks

    @property
    def device_name(self) -> str:
        if self._test_mode:
            return "Test Tone (synthetic)"
        if self._pa and self.config.audio_device_index is not None:
            info = self._pa.get_device_info_by_index(self.config.audio_device_index)
            return info.get("name", "Unknown")
        return "Default Microphone"

    def _capture_loop(self) -> None:
        chunk_interval = AUDIO_CHUNK_SAMPLES / self.config.audio_sample_rate

        while self._running:
            t0 = time.monotonic()

            if self._test_mode:
                data = self._generate_test_audio()
            else:
                try:
                    data = self._stream.read(AUDIO_CHUNK_SAMPLES, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.001)
                    continue

            with self._lock:
                self._buffer.append(data)
                max_chunks = self.config.audio_sample_rate // AUDIO_CHUNK_SAMPLES
                if len(self._buffer) > max_chunks:
                    self._buffer = self._buffer[-max_chunks:]

            if self._test_mode:
                elapsed = time.monotonic() - t0
                sleep_time = chunk_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _generate_test_audio(self) -> bytes:
        """Generate a synthetic 440Hz sine wave test tone."""
        freq = 440.0
        t = np.arange(self._test_sample_idx, self._test_sample_idx + AUDIO_CHUNK_SAMPLES)
        self._test_sample_idx += AUDIO_CHUNK_SAMPLES
        samples = (3000 * np.sin(2 * np.pi * freq * t / self.config.audio_sample_rate)).astype(np.int16)
        return samples.tobytes()


@dataclass
class _FPSCounter:
    """Simple FPS counter using a sliding window."""

    _timestamps: list[float] = field(default_factory=list)
    _window: float = 1.0  # 1 second window

    def tick(self) -> None:
        now = time.monotonic()
        self._timestamps.append(now)
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return len(self._timestamps) / self._window
