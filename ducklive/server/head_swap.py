"""Head swap engine using LivePortrait (face reenactment / motion transfer).

Pipeline:
    1. Source image: extract appearance features (F) + keypoints (M)
    2. Each driving frame: detect face → extract motion (M) → compute relative motion
    3. Warp source appearance with driving motion (W) → decode (G)
    4. Stitch + paste back onto original frame

This replaces the old InsightFace inswapper approach with full head reenactment.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Resolve LivePortrait source path — works both in dev (repo) and pip-installed
_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # ducklive/
_PROJECT_ROOT = _PACKAGE_DIR.parent

# When pip-installed, liveportrait_src/src is mapped to ducklive/_lp_src/
_LP_INSTALLED = _PACKAGE_DIR / "_lp_src"
# When running from repo, liveportrait_src/src is next to ducklive/
_LP_DEV = _PROJECT_ROOT / "liveportrait_src" / "src"
_LP_SRC = _LP_INSTALLED if _LP_INSTALLED.exists() else _LP_DEV


def _ensure_lp_path():
    """Make LivePortrait importable as ``from src.xxx import ...``."""
    if "src" in sys.modules:
        return
    if _LP_SRC == _LP_INSTALLED:
        # pip-installed: ducklive/_lp_src is the package — alias it as "src"
        import importlib
        _lp = importlib.import_module("ducklive._lp_src")
        sys.modules["src"] = _lp
        # Also alias sub-packages so "from src.config" etc. work
        for sub in ("config", "modules", "utils"):
            sub_mod = importlib.import_module(f"ducklive._lp_src.{sub}")
            sys.modules[f"src.{sub}"] = sub_mod
    else:
        # Dev mode: add liveportrait_src/ to sys.path
        lp_path = str(_LP_SRC.parent)
        if lp_path not in sys.path:
            sys.path.insert(0, lp_path)


def _make_face_mask(size: int = 512) -> np.ndarray:
    """Create an elliptical face mask (3-channel, uint8) for paste-back.

    The default mask_template.png is a white rectangle that leaks the
    generator's white background. This elliptical mask covers the full
    head region and tapers smoothly with heavy feathering.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    # Face sits slightly above center in the crop
    center = (size // 2, int(size * 0.46))
    # Large ellipse covering full head: ~72% width, ~82% height
    axes = (int(size * 0.36), int(size * 0.41))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    # Erode to create a buffer zone, preventing hard edges
    erode_k = max(size // 40, 3)  # ~12 for 512
    mask = cv2.erode(mask, np.ones((erode_k, erode_k), np.uint8), iterations=1)
    # Very heavy Gaussian blur for maximum feathering
    ksize = size // 2 | 1  # ~255 for 512, much larger than before
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return cv2.merge([mask, mask, mask])


# Pre-built elliptical mask at crop resolution (512x512)
_FACE_MASK_512: np.ndarray | None = None


def _get_face_mask(size: int = 512) -> np.ndarray:
    """Get (or lazily build) the elliptical face mask."""
    global _FACE_MASK_512
    if _FACE_MASK_512 is None or _FACE_MASK_512.shape[0] != size:
        _FACE_MASK_512 = _make_face_mask(size)
    return _FACE_MASK_512


def _exclude_white_bg(generated: np.ndarray, base_mask: np.ndarray, crop_size: int) -> np.ndarray:
    """Modulate base mask to exclude white generator background.

    The LivePortrait generator produces white/light-gray background around
    the face. If the mask extends into those areas, white bleeds into the
    composite. This detects near-white regions and reduces the mask there.

    Args:
        generated: RGB uint8, 256x256 (generator output I_p)
        base_mask: uint8 3-channel mask at crop_size resolution
        crop_size: target mask resolution (e.g. 512)

    Returns:
        Modulated uint8 3-channel mask at crop_size resolution.
    """
    gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Smooth ramp: 1.0 for pixels < 235, 0.0 for pixels > 252
    content_factor = np.clip((252.0 - gray) / 17.0, 0.0, 1.0)
    # Resize to crop_size
    if content_factor.shape[0] != crop_size:
        content_factor = cv2.resize(
            content_factor, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR
        )
    # Smooth the content mask to avoid sharp transitions
    k = crop_size // 10 | 1  # ~51 for 512
    content_factor = cv2.GaussianBlur(content_factor, (k, k), 0)
    # Apply to base mask
    mask_f = base_mask.astype(np.float32) / 255.0
    if mask_f.ndim == 3:
        content_3ch = np.stack([content_factor] * 3, axis=-1)
        mask_f = mask_f * content_3ch
    else:
        mask_f = mask_f * content_factor
    return (mask_f * 255.0).astype(np.uint8)


class HeadSwapEngine:
    """Real-time head swap engine powered by LivePortrait.

    Uses LivePortrait's motion transfer approach:
    - Source image provides appearance (feature volume via F network)
    - Driving video frames provide motion (keypoints via M network)
    - Relative motion is computed and applied to generate new face
    """

    def __init__(self, weights_dir: Path | str | None = None):
        if weights_dir is None:
            weights_dir = Path("pretrained_weights")
        self.weights_dir = Path(weights_dir)

        # LivePortrait components
        self._wrapper = None  # LivePortraitWrapper
        self._cropper = None  # Cropper

        # Source (target face to wear)
        self._source_prepared = None  # torch.Tensor (1x3x256x256)
        self._source_feature_3d = None  # torch.Tensor from F
        self._source_kp_info = None  # dict from M
        self._source_rotation = None  # rotation matrix
        self._source_x_s = None  # transformed source keypoints
        self._source_lmk = None  # 203-point landmarks for retargeting
        self._target_image_path: str = ""

        # Driving state (for relative motion)
        self._driving_first_kp_info = None  # first driving frame kp_info
        self._driving_first_rotation = None  # first driving frame R
        self._prev_lmk = None  # previous driving frame landmarks for tracking

        self._loaded = False
        self._avg_latency_ms = 0.0
        self._latency_alpha = 0.1  # EMA smoothing
        self._frame_count = 0

    def load(self) -> None:
        """Load all LivePortrait models (F, M, W, G, S)."""
        _ensure_lp_path()

        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.cropper import Cropper

        lp_weights = self.weights_dir / "liveportrait"
        insightface_root = self.weights_dir / "insightface"

        # Configure inference
        inf_cfg = InferenceConfig()
        inf_cfg.flag_use_half_precision = False
        inf_cfg.flag_do_torch_compile = False

        # MPS (Apple Silicon) produces corrupted output with LivePortrait models.
        # Force CPU on macOS; CUDA on Windows/Linux works fine.
        import platform
        if platform.system() == "Darwin":
            inf_cfg.flag_force_cpu = True
            logger.info("macOS detected — using CPU for LivePortrait (MPS not compatible)")
        else:
            inf_cfg.flag_force_cpu = False
            inf_cfg.flag_use_half_precision = True  # Enable fp16 on CUDA for speed

        # Override checkpoint paths to our weights location
        inf_cfg.checkpoint_F = str(lp_weights / "base_models" / "appearance_feature_extractor.pth")
        inf_cfg.checkpoint_M = str(lp_weights / "base_models" / "motion_extractor.pth")
        inf_cfg.checkpoint_G = str(lp_weights / "base_models" / "spade_generator.pth")
        inf_cfg.checkpoint_W = str(lp_weights / "base_models" / "warping_module.pth")
        inf_cfg.checkpoint_S = str(lp_weights / "retargeting_models" / "stitching_retargeting_module.pth")

        # models.yaml is in liveportrait_src/src/config/
        inf_cfg.models_config = str(_LP_SRC / "config" / "models.yaml")

        # Load mask_crop from resources
        mask_path = _LP_SRC / "utils" / "resources" / "mask_template.png"
        if mask_path.exists():
            inf_cfg.mask_crop = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)

        logger.info(f"Loading LivePortrait models from {lp_weights}...")
        self._wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
        logger.info(f"LivePortrait models loaded on device: {self._wrapper.device}")

        # Configure cropper
        crop_cfg = CropConfig()
        crop_cfg.insightface_root = str(insightface_root)
        crop_cfg.landmark_ckpt_path = str(lp_weights / "landmark.onnx")

        logger.info("Loading face detection models...")
        self._cropper = Cropper(crop_cfg=crop_cfg)
        logger.info("Face detection models ready")

        self._loaded = True

    def set_target_face(self, image_path: str | Path) -> bool:
        """Set the source face image (the face we want to paste/animate).

        This extracts appearance features and keypoints from the source image.
        Only needs to be done once per identity.

        Returns True if a face was successfully detected and processed.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        from src.utils.camera import get_rotation_matrix

        image_path = str(image_path)
        img_rgb = cv2.imread(image_path)
        if img_rgb is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Crop and align source face
        crop_cfg = self._cropper.crop_cfg
        crop_info = self._cropper.crop_source_image(img_rgb, crop_cfg)
        if crop_info is None:
            logger.warning(f"No face detected in source image: {image_path}")
            return False

        img_crop_256 = crop_info["img_crop_256x256"]
        self._source_lmk = crop_info.get("lmk_crop")

        # Prepare source tensor
        I_s = self._wrapper.prepare_source(img_crop_256)

        # Extract appearance feature volume (F network)
        self._source_feature_3d = self._wrapper.extract_feature_3d(I_s)

        # Extract motion keypoints (M network)
        self._source_kp_info = self._wrapper.get_kp_info(I_s)
        self._source_rotation = get_rotation_matrix(
            self._source_kp_info['pitch'],
            self._source_kp_info['yaw'],
            self._source_kp_info['roll']
        )
        self._source_x_s = self._wrapper.transform_keypoint(self._source_kp_info)

        # Store crop info for paste-back
        self._source_crop_info = crop_info
        self._source_img_rgb = img_rgb

        # Reset driving state
        self._driving_first_kp_info = None
        self._driving_first_rotation = None
        self._prev_lmk = None
        self._frame_count = 0

        self._target_image_path = image_path
        logger.info(f"Target face set: {image_path}")
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single BGR video frame — detect driving face and render head swap.

        Args:
            frame: BGR frame from camera/video (any resolution)

        Returns:
            BGR frame with swapped head, or original frame if processing fails.
        """
        if not self._loaded or self._source_feature_3d is None:
            return frame

        t0 = time.perf_counter()

        try:
            result = self._process_frame_inner(frame)
        except Exception as e:
            logger.debug(f"Frame processing failed: {e}")
            return frame

        # Update latency EMA
        latency_ms = (time.perf_counter() - t0) * 1000
        self._avg_latency_ms = (
            self._latency_alpha * latency_ms
            + (1 - self._latency_alpha) * self._avg_latency_ms
        )
        self._frame_count += 1

        return result

    def _process_frame_inner(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Core frame processing logic."""
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import prepare_paste_back, paste_back

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Step 1: Detect & crop driving face ---
        driving_crop = self._crop_driving_face(frame_rgb)
        if driving_crop is None:
            return frame_bgr

        driving_img_256 = driving_crop["img_crop_256x256"]
        driving_lmk = driving_crop.get("lmk_crop")
        driving_M_c2o = driving_crop["M_c2o"]

        # --- Step 2: Extract driving motion ---
        I_d = self._wrapper.prepare_source(driving_img_256)
        driving_kp_info = self._wrapper.get_kp_info(I_d)
        driving_rotation = get_rotation_matrix(
            driving_kp_info['pitch'],
            driving_kp_info['yaw'],
            driving_kp_info['roll']
        )

        # Cache first frame for relative motion
        if self._driving_first_kp_info is None:
            self._driving_first_kp_info = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in driving_kp_info.items()
            }
            self._driving_first_rotation = driving_rotation.clone()

        # --- Step 3: Compute relative motion (source image + driving delta) ---
        x_s_info = self._source_kp_info
        x_c_s = x_s_info['kp']
        R_s = self._source_rotation
        x_s = self._source_x_s

        # Relative rotation: R_new = (R_d_i @ R_d_0^T) @ R_s
        R_d_i = driving_rotation
        R_d_0 = self._driving_first_rotation
        R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s

        # Relative expression: delta_new = source_exp + (driving_exp - driving_first_exp)
        x_d_0_info = self._driving_first_kp_info
        delta_new = x_s_info['exp'] + (driving_kp_info['exp'] - x_d_0_info['exp'])

        # Relative scale and translation
        scale_new = x_s_info['scale'] * (driving_kp_info['scale'] / x_d_0_info['scale'])
        t_new = x_s_info['t'] + (driving_kp_info['t'] - x_d_0_info['t'])
        t_new[..., 2].fill_(0)  # zero tz

        # Compute new driving keypoints: x_d_new = s * (kp @ R + exp) + t
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        # --- Step 4: Stitching ---
        if self._wrapper.stitching_retargeting_module is not None:
            x_d_i_new = self._wrapper.stitching(x_s, x_d_i_new)

        # Apply driving multiplier (default 1.0)
        x_d_i_new = x_s + (x_d_i_new - x_s) * self._wrapper.inference_cfg.driving_multiplier

        # --- Step 5: Warp + Decode ---
        out = self._wrapper.warp_decode(self._source_feature_3d, x_s, x_d_i_new)
        I_p = self._wrapper.parse_output(out['out'])[0]  # 256x256x3, RGB, uint8

        # --- Step 6: Paste back with content-aware face mask ---
        # Use an elliptical base mask modulated by content detection to
        # exclude the generator's white background from the composite.
        crop_size = self._cropper.crop_cfg.dsize
        base_mask = _get_face_mask(crop_size)
        face_mask = _exclude_white_bg(I_p, base_mask, crop_size)
        mask_ori_float = prepare_paste_back(
            face_mask,
            driving_M_c2o,
            dsize=(frame_rgb.shape[1], frame_rgb.shape[0])
        )

        result_rgb = paste_back(I_p, driving_M_c2o, frame_rgb, mask_ori_float)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    def _crop_driving_face(self, frame_rgb: np.ndarray) -> dict | None:
        """Detect and crop the driving face from frame.

        Uses InsightFace for initial detection, then landmark.onnx for
        precise 203-point landmarks. Tracking: uses previous landmarks
        for subsequent frames to avoid re-detection overhead.
        """
        from src.utils.crop import crop_image

        crop_cfg = self._cropper.crop_cfg

        if self._prev_lmk is None:
            # First frame or lost tracking: full detection
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            src_face = self._cropper.face_analysis_wrapper.get(
                frame_bgr,
                flag_do_landmark_2d_106=True,
                direction=crop_cfg.direction,
            )
            if len(src_face) == 0:
                return None

            src_face = src_face[0]
            lmk = src_face.landmark_2d_106
            # Refine with landmark runner
            lmk = self._cropper.human_landmark_runner.run(frame_rgb, lmk)
        else:
            # Tracking mode: refine from previous landmarks
            try:
                lmk = self._cropper.human_landmark_runner.run(frame_rgb, self._prev_lmk)
            except Exception:
                self._prev_lmk = None
                return None

        self._prev_lmk = lmk

        # Crop the face region
        ret_dct = crop_image(
            frame_rgb,
            lmk,
            dsize=crop_cfg.dsize,
            scale=crop_cfg.scale,
            vx_ratio=crop_cfg.vx_ratio,
            vy_ratio=crop_cfg.vy_ratio,
            flag_do_rot=crop_cfg.flag_do_rot,
        )

        ret_dct["img_crop_256x256"] = cv2.resize(
            ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
        )
        ret_dct["lmk_crop"] = lmk

        return ret_dct

    def reset_driving(self) -> None:
        """Reset the driving state (first-frame cache).

        Call this when the driving person changes or tracking is lost.
        """
        self._driving_first_kp_info = None
        self._driving_first_rotation = None
        self._prev_lmk = None
        self._frame_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_target(self) -> bool:
        return self._source_feature_3d is not None

    @property
    def target_image_path(self) -> str:
        return self._target_image_path

    @property
    def avg_latency_ms(self) -> float:
        return self._avg_latency_ms

    @property
    def model_name(self) -> str:
        return "liveportrait" if self._loaded else ""
