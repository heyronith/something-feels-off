"""Git hash, package versions, model revision, seeds."""

from __future__ import annotations

import os
import random
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch


def git_hash(cwd: Optional[str] = None) -> str:
    """Return current HEAD commit hash, or 'unknown' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return os.environ.get("SFO_GIT_HASH", "unknown")


def set_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def package_versions() -> Dict[str, str]:
    import transformers

    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    try:
        import accelerate

        versions["accelerate"] = accelerate.__version__
    except ImportError:
        pass
    try:
        import huggingface_hub

        versions["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        pass
    return versions


def cuda_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda": None,
        "gpu_name": None,
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def collect_env() -> Dict[str, Any]:
    versions = package_versions()
    cuda = cuda_env()
    return {
        "python": versions["python"],
        "torch": versions["torch"],
        "transformers": versions["transformers"],
        "cuda": cuda["cuda"],
        "gpu_name": cuda["gpu_name"],
    }
