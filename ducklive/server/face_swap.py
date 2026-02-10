"""Face swap engine using InsightFace inswapper model.

Pipeline:
    1. Detect faces in source frame (RetinaFace via InsightFace)
    2. Extract face embedding from target image (one-time)
    3. Swap detected face with target face (inswapper_128)
    4. Blend result back into frame
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _detect_onnx_providers() -> list[str]:
    """Auto-detect the best available ONNX Runtime execution providers.

    Priority: CUDA > CoreML (Apple Silicon) > CPU
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        logger.info("🖥️  Detected NVIDIA GPU — using CUDA acceleration")
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
        logger.info("🍎 Detected Apple Silicon — using CoreML acceleration")
    providers.append("CPUExecutionProvider")  # Always include CPU as fallback

    if len(providers) == 1:
        logger.warning("⚠️  No GPU detected — face swap will run on CPU (slower)")

    return providers


class FaceSwapEngine:
    """Real-time face swap engine powered by InsightFace."""

    def __init__(self, model_dir: Path | str = "models"):
        self.model_dir = Path(model_dir)
        self._analyser = None
        self._swapper = None
        self._target_face = None  # Cached target face embedding
        self._target_image_path: str = ""
        self._loaded = False
        self._avg_latency_ms = 0.0
        self._latency_alpha = 0.1  # EMA smoothing

    def load(self, providers: list[str] | None = None) -> None:
        """Load face detection and swap models.

        Auto-detects available execution providers:
          - NVIDIA GPU → CUDAExecutionProvider
          - Apple Silicon → CoreMLExecutionProvider
          - Fallback → CPUExecutionProvider

        Args:
            providers: Override execution providers list. Auto-detected if None.
        """
        import insightface
        from insightface.app import FaceAnalysis

        if providers is None:
            providers = _detect_onnx_providers()

        logger.info(f"Face swap using providers: {providers}")

        # Face detector + embedder (buffalo_l auto-downloaded to ~/.insightface/models/)
        self._analyser = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self._analyser.prepare(ctx_id=0, det_size=(640, 640))

        # Face swapper model
        model_path = self.model_dir / "inswapper_128.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"inswapper model not found at {model_path}. "
                "Download from: https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
            )
        self._swapper = insightface.model_zoo.get_model(
            str(model_path), providers=providers
        )
        self._loaded = True

    def set_target_face(self, image_path: str | Path) -> bool:
        """Set the target face from an image file.

        Returns True if a face was successfully detected.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        faces = self._analyser.get(image)
        if not faces:
            return False

        # Use the largest face
        self._target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self._target_image_path = str(image_path)
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame — detect and swap faces.

        Returns the frame with face(s) swapped. If no face detected or
        no target set, returns original frame unchanged.
        """
        if not self._loaded or self._target_face is None:
            return frame

        t0 = time.perf_counter()

        # Detect faces in current frame
        source_faces = self._analyser.get(frame)
        if not source_faces:
            return frame

        result = frame.copy()
        # Swap the largest face (primary subject)
        source_face = max(
            source_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        result = self._swapper.get(result, source_face, self._target_face, paste_back=True)

        # Update latency EMA
        latency_ms = (time.perf_counter() - t0) * 1000
        self._avg_latency_ms = (
            self._latency_alpha * latency_ms + (1 - self._latency_alpha) * self._avg_latency_ms
        )

        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_target(self) -> bool:
        return self._target_face is not None

    @property
    def target_image_path(self) -> str:
        return self._target_image_path

    @property
    def avg_latency_ms(self) -> float:
        return self._avg_latency_ms

    @property
    def model_name(self) -> str:
        return "inswapper_128" if self._loaded else ""
