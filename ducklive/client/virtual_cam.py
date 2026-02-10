"""Virtual camera output — pipes processed video to a virtual webcam device.

On macOS: requires OBS Virtual Camera (installed with OBS) or
          a custom CoreMediaIO DAL plugin.
On Windows: uses OBS Virtual Camera or DirectShow.

The virtual camera appears as "DuckLive Camera" in Zoom, Teams, etc.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VirtualCamera:
    """Output video frames to a virtual camera device."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._cam = None
        self._running = False

    def start(self) -> None:
        """Start the virtual camera device."""
        try:
            import pyvirtualcam

            self._cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                device="DuckLive Camera",
            )
            self._running = True
            logger.info(
                f"Virtual camera started: {self._cam.device} "
                f"({self.width}x{self.height}@{self.fps}fps)"
            )
        except ImportError:
            logger.error(
                "pyvirtualcam not installed. Install with: pip install pyvirtualcam\n"
                "On macOS, also install OBS for the virtual camera backend."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to start virtual camera: {e}")
            raise

    def stop(self) -> None:
        """Stop the virtual camera."""
        self._running = False
        if self._cam:
            self._cam.close()
            self._cam = None
            logger.info("Virtual camera stopped")

    def send_jpeg(self, jpeg_data: bytes) -> None:
        """Decode a JPEG frame and send it to the virtual camera."""
        if not self._cam or not self._running:
            return

        # Decode JPEG to numpy array
        nparr = np.frombuffer(jpeg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Resize if needed
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self._cam.send(frame)

    def send_frame(self, frame: np.ndarray) -> None:
        """Send a raw BGR numpy frame to the virtual camera."""
        if not self._cam or not self._running:
            return

        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self._cam.send(frame)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device_name(self) -> str:
        return self._cam.device if self._cam else "N/A"
