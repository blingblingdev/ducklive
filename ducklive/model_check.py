"""Pre-flight model check — verify required models exist before server start."""

from __future__ import annotations

import sys
from pathlib import Path

# Models that MUST exist for the server to start
REQUIRED_MODELS = {
    "models/inswapper_128.onnx": {
        "desc": "InsightFace face swap model",
        "url": "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
    },
    "models/hubert_base.pt": {
        "desc": "HuBERT base model for voice feature extraction",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
    },
    "models/rmvpe.pt": {
        "desc": "RMVPE pitch extraction model",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    },
}

# LivePortrait weights — needed for the primary face swap engine
LIVEPORTRAIT_WEIGHTS = {
    "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth": {
        "desc": "LivePortrait appearance feature extractor (F)",
    },
    "pretrained_weights/liveportrait/base_models/motion_extractor.pth": {
        "desc": "LivePortrait motion extractor (M)",
    },
    "pretrained_weights/liveportrait/base_models/spade_generator.pth": {
        "desc": "LivePortrait generator (G)",
    },
    "pretrained_weights/liveportrait/base_models/warping_module.pth": {
        "desc": "LivePortrait warping module (W)",
    },
    "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth": {
        "desc": "LivePortrait stitching module (S)",
    },
    "pretrained_weights/liveportrait/landmark.onnx": {
        "desc": "LivePortrait landmark detector",
    },
    "pretrained_weights/insightface/models/buffalo_l/det_10g.onnx": {
        "desc": "InsightFace face detector (buffalo_l)",
    },
}

DOWNLOAD_INSTRUCTIONS = """\

Required model files are missing. Place them in the current directory:

  {cwd}/

MISSING FILES:
{missing_list}

DOWNLOAD INSTRUCTIONS:

  1. Core models (models/):
     Download from HuggingFace and place in ./models/:
{model_urls}

  2. LivePortrait weights (pretrained_weights/):
     git clone https://huggingface.co/KwaiVGI/LivePortrait pretrained_weights/liveportrait
     git clone https://huggingface.co/KwaiVGI/LivePortrait-Animals pretrained_weights/liveportrait_animals

  3. InsightFace buffalo_l (pretrained_weights/insightface/):
     Download from https://github.com/deepinsight/insightface/releases
     Place .onnx files in pretrained_weights/insightface/models/buffalo_l/

  4. Voice models (voices/) — optional:
     Place RVC .pth voice model files in ./voices/

  5. Face images (faces/) — optional:
     Place target face .jpg/.png images in ./faces/

After downloading, run `ducklive server` again.
"""


def check_models(strict: bool = False) -> bool:
    """Check that required models exist in the current working directory.

    Args:
        strict: If True, exit with error if any models are missing.
                If False, print warnings but continue.

    Returns:
        True if all models are present, False otherwise.
    """
    cwd = Path.cwd()
    missing = []

    # Check required models
    for rel_path, info in REQUIRED_MODELS.items():
        full_path = cwd / rel_path
        if not full_path.exists():
            missing.append((rel_path, info))

    # Check LivePortrait weights
    for rel_path, info in LIVEPORTRAIT_WEIGHTS.items():
        full_path = cwd / rel_path
        if not full_path.exists():
            missing.append((rel_path, info))

    if not missing:
        return True

    # Build missing list
    missing_list = "\n".join(f"  - {path}  ({info['desc']})" for path, info in missing)

    # Build download URLs for models that have them
    model_urls = ""
    for path, info in missing:
        if "url" in info:
            model_urls += f"     {info['url']}\n       -> {path}\n"

    msg = DOWNLOAD_INSTRUCTIONS.format(
        cwd=cwd,
        missing_list=missing_list,
        model_urls=model_urls.rstrip() if model_urls else "     (see instructions above)",
    )

    if strict:
        print(msg, file=sys.stderr)
        sys.exit(1)
    else:
        # Print warning but continue — engines will fail gracefully at load time
        from rich.console import Console
        console = Console(stderr=True)
        console.print(f"[yellow bold]WARNING: {len(missing)} model file(s) not found.[/]")
        console.print("[yellow]Some engines will be unavailable. Run with --check-models for details.[/]")
        return False
