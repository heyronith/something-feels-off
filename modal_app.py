"""Modal app: image, volume, secrets, WO-000 functions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

APP_NAME = "something-feels-off"
MODEL_ID = "google/gemma-3-27b-it"
GEMMASCOPE_REPO = "google/gemma-scope-2-27b-it"
VOLUME_NAME = "sfo-models"
HF_SECRET_NAME = "hf-token"  # env key inside secret: HF_TOKEN
MODEL_DIR = "/models/gemma-3-27b-it"

# Explicit resource requests for budget ledger (per configs/wo_000.yaml).
GPU_CPU_CORES = 8.0
GPU_MEMORY_MIB = 65536  # 64 GiB
GPU_MEMORY_GIB = GPU_MEMORY_MIB / 1024.0

REPO = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(
        str(REPO / "requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .env(
        {
            "HF_HOME": "/models/hf_home",
            "TRANSFORMERS_CACHE": "/models/hf_home",
            "PYTHONPATH": "/root/src",
        }
    )
    .add_local_dir(
        str(REPO / "src"),
        remote_path="/root/src",
        ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
    )
    .add_local_dir(
        str(REPO / "scripts"),
        remote_path="/root/scripts",
        ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
    )
    .add_local_dir(str(REPO / "configs"), remote_path="/root/configs")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


def _hf_token() -> str:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            f"HF_TOKEN missing from Modal secret {HF_SECRET_NAME!r}"
        )
    return token


@app.function(
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=60 * 60 * 4,
    cpu=4.0,
    memory=16384,
)
def download_model() -> dict:
    """CPU-only: snapshot_download google/gemma-3-27b-it to the volume."""
    from huggingface_hub import HfApi, snapshot_download

    token = _hf_token()
    api = HfApi(token=token)
    info = api.model_info(MODEL_ID)
    sha = info.sha
    print(f"Resolved {MODEL_ID} revision SHA: {sha}")

    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=sha,
        local_dir=MODEL_DIR,
        token=token,
    )
    volume.commit()
    return {"model_id": MODEL_ID, "revision_sha": sha, "path": path}


@app.function(
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=30 * 60,
    cpu=2.0,
    memory=4096,
)
def list_gemmascope_manifest() -> str:
    """CPU-only: list files under transcoder_all/ (no weight download)."""
    from huggingface_hub import list_repo_files

    token = _hf_token()
    files = list_repo_files(GEMMASCOPE_REPO, repo_type="model", token=token)
    prefix = "transcoder_all/"
    listed = sorted(f for f in files if f.startswith(prefix))
    text = "\n".join(listed) + ("\n" if listed else "")
    print(f"Found {len(listed)} files under {prefix}")
    return text


@app.function(timeout=30 * 60, cpu=2.0, memory=4096)
def pip_freeze() -> str:
    """Return full pip freeze from inside the Modal image."""
    return subprocess.check_output(["pip", "freeze"], text=True)


@app.function(
    gpu="A100-80GB",
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=45 * 60,
    cpu=GPU_CPU_CORES,
    memory=GPU_MEMORY_MIB,
)
def smoke_test(git_hash: str, revision_sha: str) -> dict:
    """GPU smoke test for WO-000. Single container; model from volume."""
    import importlib.util
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    results_dir = Path("/tmp/results/WO-000")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)
    ledger_path = Path("/tmp/results/budget_ledger.csv")
    log_path = results_dir / "raw" / "smoke_log.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    spec = importlib.util.spec_from_file_location(
        "wo_000_smoke", "/root/scripts/wo_000_smoke.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    start = datetime.now(timezone.utc)
    with open(log_path, "w") as logf:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, logf)
        sys.stderr = Tee(old_err, logf)
        try:
            spec.loader.exec_module(mod)
            metrics = mod.run_smoke(
                model_dir=MODEL_DIR,
                revision_sha=revision_sha,
                config_path="/root/configs/wo_000.yaml",
                results_dir=str(results_dir),
                ledger_path=str(ledger_path),
                git_hash_str=git_hash,
                gpu_type="A100-80GB",
                n_cores=GPU_CPU_CORES,
                mem_gib=GPU_MEMORY_GIB,
                start_utc=start,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

    # Return only text artifacts so the local entrypoint does not need torch.
    artifacts = {
        "metrics.json": (results_dir / "metrics.json").read_text(),
        "smoke_log.txt": log_path.read_text(),
        "budget_ledger.csv": ledger_path.read_text()
        if ledger_path.exists()
        else "",
    }
    # Compact pass/fail summary (no tensors / non-JSON types)
    summary = {
        name: bool(c.get("pass")) for name, c in (metrics or {}).get("checks", {}).items()
    }
    return {"summary": summary, "artifacts": artifacts}


def _read_revision_sha(rev_path: Path) -> str | None:
    """Parse revision_sha from model_revision.yaml without PyYAML (local entrypoint)."""
    if not rev_path.exists():
        return None
    for line in rev_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("revision_sha:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


@app.local_entrypoint()
def main(action: str = "help"):
    """
    Actions:
      freeze | download | manifest | smoke | all
    """
    root = Path(__file__).resolve().parent
    results = root / "results" / "WO-000"
    results.mkdir(parents=True, exist_ok=True)
    (results / "raw").mkdir(parents=True, exist_ok=True)

    if action == "help":
        print(
            "Usage: modal run modal_app.py --action [freeze|download|manifest|smoke|all]"
        )
        return

    if action in ("freeze", "all"):
        print("=== pip freeze ===")
        freeze = pip_freeze.remote()
        lock_path = root / "requirements.lock"
        lock_path.write_text(freeze)
        print(f"Wrote {lock_path} ({len(freeze.splitlines())} lines)")

    rev_path = root / "configs" / "model_revision.yaml"
    revision_sha = _read_revision_sha(rev_path)

    if action in ("download", "all"):
        print("=== download_model ===")
        info = download_model.remote()
        revision_sha = info["revision_sha"]
        rev_path.write_text(
            "# Pinned by WO-000 download_model via HfApi().model_info\n"
            f"model_id: {MODEL_ID}\n"
            f"revision_sha: {revision_sha}\n"
        )
        print(f"Wrote {rev_path}: {revision_sha}")

    if action in ("manifest", "all"):
        print("=== gemmascope manifest ===")
        text = list_gemmascope_manifest.remote()
        man_path = results / "gemmascope2_manifest.txt"
        man_path.write_text(text)
        print(f"Wrote {man_path} ({len(text.splitlines())} files)")

    if action in ("smoke", "all"):
        if not revision_sha:
            raise SystemExit(
                "No revision_sha — run download first or ensure "
                "configs/model_revision.yaml exists"
            )
        print("=== smoke_test ===")
        gh = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
        ).strip()
        out = smoke_test.remote(git_hash=gh, revision_sha=revision_sha)
        arts = out["artifacts"]
        (results / "metrics.json").write_text(arts["metrics.json"])
        (results / "raw" / "smoke_log.txt").write_text(arts["smoke_log.txt"])
        ledger = root / "results" / "budget_ledger.csv"
        remote_csv = arts["budget_ledger.csv"]
        if remote_csv:
            remote_lines = remote_csv.strip().splitlines()
            if not ledger.exists() or ledger.stat().st_size == 0:
                ledger.write_text(remote_csv if remote_csv.endswith("\n") else remote_csv + "\n")
            else:
                with ledger.open("a") as f:
                    for line in remote_lines[1:]:
                        f.write(line + "\n")
        print("Smoke artifacts written to results/WO-000/")
        for name, ok in out.get("summary", {}).items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        metrics_path = results / "metrics.json"
        if metrics_path.exists():
            import json

            m = json.loads(metrics_path.read_text())
            failed = [k for k, v in m.get("checks", {}).items() if not v.get("pass")]
            if failed:
                print(f"Failed checks: {failed}")
            else:
                print("All smoke checks PASS")
