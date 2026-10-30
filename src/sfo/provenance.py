"""Git hash, package versions, model revision, seeds."""

from __future__ import annotations

import os
import random
import subprocess
from typing import Any, Dict, Optional


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


def assert_clean_pushed_main(cwd: Optional[str] = None) -> str:
    """
    Assert a clean working tree and that HEAD exists on origin/main.

    Returns the HEAD hash to use for run logging. Aborts (raises) otherwise.
    Does not import torch/numpy (safe for Modal local entrypoint).
    """
    root = cwd or os.getcwd()

    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
    )
    if status.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit or stash changes before launching "
            f"GPU runs.\n{status}"
        )

    subprocess.check_call(
        ["git", "fetch", "origin", "main"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    origin_main = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=root, text=True
    ).strip()

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", head, "origin/main"],
        cwd=root,
    )
    if rc != 0:
        raise RuntimeError(
            f"HEAD {head} is not present on origin/main ({origin_main}). "
            "Push before launching GPU runs."
        )

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True
    ).strip()
    if branch != "main":
        raise RuntimeError(
            f"Expected branch main, got {branch!r}. Switch to main before launching."
        )

    print(f"[provenance] clean tree; HEAD {head} is on origin/main")
    return head


def set_seeds(seed: int = 0) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def package_versions() -> Dict[str, str]:
    import torch
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
    import torch

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
