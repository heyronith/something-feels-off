"""WO-003: download GemmaScope-2 27B-IT transcoders (layers 45, 50; 262k, big, affine).

Only config.json + params.safetensors per layer; examples.safetensors is never fetched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

REPO_ID = "google/gemma-scope-2-27b-it"
LAYERS = (45, 50)
SUBDIR = "gemmascope2-27b-it"
FILES = ("config.json", "params.safetensors")


def _folder(layer: int) -> str:
    return f"transcoder_all/layer_{layer}_width_262k_l0_big_affine"


def run_download(volume_dir: str, repo_id: str = REPO_ID) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    target = Path(volume_dir) / SUBDIR
    target.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(repo_id)
    sha = info.sha

    allow: List[str] = [f"{_folder(l)}/{n}" for l in LAYERS for n in FILES]
    assert not any("examples" in p for p in allow)

    paths: Dict[str, str] = {}
    for rel in allow:
        # Pin to the recorded sha so all files come from the same revision.
        p = hf_hub_download(repo_id=repo_id, filename=rel, revision=sha, local_dir=str(target))
        paths[rel] = str(p)

    sizes = {rel: Path(p).stat().st_size for rel, p in paths.items()}
    result = {
        "repo_id": repo_id,
        "revision": sha,
        "downloaded_at_unix": time.time(),
        "target_dir": str(target),
        "files": paths,
        "sizes_bytes": sizes,
        "layer_dirs": {str(l): str(target / _folder(l)) for l in LAYERS},
    }
    (target / "revision.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--volume-dir", default="/models")
    ap.add_argument("--repo-id", default=REPO_ID)
    a = ap.parse_args()
    print(json.dumps(run_download(a.volume_dir, a.repo_id), indent=2))
