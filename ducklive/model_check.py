"""Pre-flight model check — verify required models exist, auto-download if missing."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_HF_LP = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main"
_HF_RVC = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

# All models with download URLs
ALL_MODELS: dict[str, dict[str, str]] = {
    # Core models
    "models/inswapper_128.onnx": {
        "desc": "InsightFace face swap model",
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    },
    "models/hubert_base.pt": {
        "desc": "HuBERT base model for voice feature extraction",
        "url": f"{_HF_RVC}/hubert_base.pt",
    },
    "models/rmvpe.pt": {
        "desc": "RMVPE pitch extraction model",
        "url": f"{_HF_RVC}/rmvpe.pt",
    },
    # LivePortrait weights
    "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth": {
        "desc": "LivePortrait appearance feature extractor (F)",
        "url": f"{_HF_LP}/liveportrait/base_models/appearance_feature_extractor.pth",
    },
    "pretrained_weights/liveportrait/base_models/motion_extractor.pth": {
        "desc": "LivePortrait motion extractor (M)",
        "url": f"{_HF_LP}/liveportrait/base_models/motion_extractor.pth",
    },
    "pretrained_weights/liveportrait/base_models/spade_generator.pth": {
        "desc": "LivePortrait generator (G)",
        "url": f"{_HF_LP}/liveportrait/base_models/spade_generator.pth",
    },
    "pretrained_weights/liveportrait/base_models/warping_module.pth": {
        "desc": "LivePortrait warping module (W)",
        "url": f"{_HF_LP}/liveportrait/base_models/warping_module.pth",
    },
    "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth": {
        "desc": "LivePortrait stitching module (S)",
        "url": f"{_HF_LP}/liveportrait/retargeting_models/stitching_retargeting_module.pth",
    },
    "pretrained_weights/liveportrait/landmark.onnx": {
        "desc": "LivePortrait landmark detector",
        "url": f"{_HF_LP}/liveportrait/landmark.onnx",
    },
    # InsightFace detector (bundled in LivePortrait HF repo)
    "pretrained_weights/insightface/models/buffalo_l/det_10g.onnx": {
        "desc": "InsightFace face detector (buffalo_l)",
        "url": f"{_HF_LP}/insightface/models/buffalo_l/det_10g.onnx",
    },
    "pretrained_weights/insightface/models/buffalo_l/2d106det.onnx": {
        "desc": "InsightFace 2D-106 landmark detector (buffalo_l)",
        "url": f"{_HF_LP}/insightface/models/buffalo_l/2d106det.onnx",
    },
}


def _download_file(url: str, dest: Path, desc: str) -> bool:
    """Download a single file with rich progress bar."""
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    console = Console(stderr=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ducklive/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))

            with Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(desc, total=total or None)
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 64)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        tmp.rename(dest)
        return True
    except Exception as e:
        console.print(f"[red]Failed to download {desc}: {e}[/]")
        if tmp.exists():
            tmp.unlink()
        return False


def _find_missing(cwd: Path) -> list[tuple[str, dict[str, str]]]:
    """Return list of (rel_path, info) for models not found on disk."""
    missing = []
    for rel_path, info in ALL_MODELS.items():
        if not (cwd / rel_path).exists():
            missing.append((rel_path, info))
    return missing


def check_models(strict: bool = False) -> bool:
    """Check required models, auto-download any that are missing.

    Args:
        strict: If True, exit with error on download failure.
                If False, warn but continue (engines fail gracefully later).

    Returns:
        True if all models are present after check/download.
    """
    from rich.console import Console

    console = Console(stderr=True)
    cwd = Path.cwd()
    missing = _find_missing(cwd)

    if not missing:
        return True

    console.print(
        f"\n[yellow bold]{len(missing)} model file(s) not found. Downloading...[/]\n"
    )

    failed = []
    for rel_path, info in missing:
        dest = cwd / rel_path
        url = info["url"]
        ok = _download_file(url, dest, info["desc"])
        if not ok:
            failed.append((rel_path, info))

    if failed:
        console.print(f"\n[red bold]{len(failed)} download(s) failed:[/]")
        for path, info in failed:
            console.print(f"  [red]- {path}  ({info['desc']})[/]")
            console.print(f"    [dim]{info['url']}[/]")
        console.print("\n[yellow]Download manually and place in the paths above.[/]")

        if strict:
            sys.exit(1)
        return False

    console.print("\n[green bold]All models downloaded successfully.[/]\n")
    return True
