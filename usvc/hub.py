"""Auto-download pretrained weights from ModelScope / HuggingFace.

All backends reference weights by relative path (e.g.
``pretrained/contentvec/pytorch_model.bin``).  When a file is missing,
:func:`ensure_pretrained` downloads it from the usvc-pretrained model hub
(HuggingFace ``baicai1145/usvc-pretrained``, mirrored on ModelScope).

Resolution order for the base ``pretrained/`` directory:
  1. ``USVC_PRETRAINED_DIR`` env var.
  2. ``./pretrained`` (cwd, convenient for local dev).
  3. ``<package_root>/pretrained`` (bundled with editable installs).

The hub repo layout mirrors the local directory structure exactly, so a
remote path like ``contentvec/pytorch_model.bin`` maps to
``<pretrained_dir>/contentvec/pytorch_model.bin`` locally.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_HF_REPO = "baicai1145/usvc-pretrained"
_MS_REPO = "baicai1145/usvc-pretrained"  # ModelScope uses the same path layout


def pretrained_dir() -> Path:
    """Return the base pretrained directory, creating it if necessary."""
    env = os.environ.get("USVC_PRETRAINED_DIR")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        # Prefer ./pretrained (cwd) so users can place weights next to their
        # project; fall back to the package-level directory for dev installs.
        cwd_p = Path.cwd() / "pretrained"
        pkg_p = Path(__file__).resolve().parents[1] / "pretrained"
        p = cwd_p if cwd_p.exists() else pkg_p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve(relative_path: str | Path) -> Path:
    """Resolve a ``pretrained/xxx`` path to an absolute local path.

    Strips a leading ``pretrained/`` if present, then joins with the base dir.
    """
    rel = str(relative_path).replace("\\", "/")
    if rel.startswith("pretrained/"):
        rel = rel[len("pretrained/") :]
    return pretrained_dir() / rel


def _download_hf(remote: str, local: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=_HF_REPO, filename=remote, local_dir=str(local.parent))
    # hf_hub_download saves as <local_dir>/<remote basename>; rename if needed
    src = Path(downloaded)
    if src != local and src.exists():
        shutil.move(str(src), str(local))
    return local.exists()


def _download_ms(remote: str, local: Path) -> bool:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as _ms_dl
    except ImportError:
        try:
            from modelscope import snapshot_download as _ms_dl
        except ImportError:
            return False
    # ModelScope downloads the whole repo subdir; we then move the file.
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = _ms_dl(_MS_REPO, local_dir=str(local.parent))
        # cache is the dir; find our file
        candidate = Path(cache) / remote
        if candidate.exists() and candidate != local:
            shutil.move(str(candidate), str(local))
    except Exception:
        return False
    return local.exists()


def ensure_pretrained(relative_path: str | Path) -> str:
    """Ensure a pretrained weight file exists locally, downloading if needed.

    Args:
        relative_path: path relative to ``pretrained/`` (e.g.
            ``"contentvec/pytorch_model.bin"`` or
            ``"pretrained/rmvpe/model.pt"``).

    Returns:
        Absolute path string to the local file.
    """
    local = _resolve(relative_path)
    if local.exists():
        return str(local)

    # Determine the remote path (without "pretrained/" prefix).
    rel = str(relative_path).replace("\\", "/")
    if rel.startswith("pretrained/"):
        rel = rel[len("pretrained/") :]

    # Try ModelScope first (faster in China, no proxy needed), then HuggingFace.
    print(f"[usvc] downloading {rel} from ModelScope...")
    if _download_ms(rel, local):
        print(f"[usvc] downloaded: {local}")
        return str(local)

    print(f"[usvc] trying HuggingFace...")
    if _download_hf(rel, local):
        print(f"[usvc] downloaded: {local}")
        return str(local)

    raise FileNotFoundError(
        f"Pretrained weight not found locally ({local}) and download failed.\n"
        f"Please download manually from:\n"
        f"  HF: https://huggingface.co/{_HF_REPO}\n"
        f"  MS: https://www.modelscope.cn/models/{_MS_REPO}\n"
        f"Place at: {local}"
    )


def ensure_dir(relative_dir: str | Path) -> str:
    """Ensure a directory of pretrained files exists (e.g. nsf_hifigan/).

    Downloads all files in the remote directory.
    """
    local = _resolve(relative_dir)
    if local.exists() and any(local.iterdir()):
        return str(local)

    rel = str(relative_dir).replace("\\", "/")
    if rel.startswith("pretrained/"):
        rel = rel[len("pretrained/") :]
    local.mkdir(parents=True, exist_ok=True)

    # Try MS first (faster in China)
    try:
        from modelscope import snapshot_download as _ms_dl

        _ms_dl(_MS_REPO, local_dir=str(local.parent))
        if any(local.iterdir()):
            return str(local)
    except Exception:
        pass

    # Try HF
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=_HF_REPO, allow_patterns=f"{rel}/*", local_dir=str(local.parent))
        if any(local.iterdir()):
            return str(local)
    except Exception:
        pass

    raise FileNotFoundError(
        f"Pretrained directory not found ({local}) and download failed.\n"
        f"Please download manually from {_HF_REPO} or {_MS_REPO}."
    )
