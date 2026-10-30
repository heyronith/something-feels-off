"""Modal app: image, volume, secrets, WO-000 functions."""

from __future__ import annotations

import subprocess
import sys
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
    .add_local_dir(
        str(REPO / "third_party" / "introspection-mechanisms"),
        remote_path="/root/third_party/introspection-mechanisms",
        ignore=["**/.git/**", "**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        str(REPO / "results" / "WO-001"),
        remote_path="/root/results/WO-001",
        ignore=["**/__pycache__/**"],
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
openai_secret = modal.Secret.from_name("openai-key")

VOLUME_RESULTS = "/models/results"
VOLUME_LEDGER = "/models/results/budget_ledger.csv"


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


@app.function(
    gpu="A100-80GB",
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=120 * 60,
    cpu=GPU_CPU_CORES,
    memory=GPU_MEMORY_MIB,
)
def wo_001_detection(git_hash: str, revision_sha: str) -> dict:
    """GPU: sanity, vectors, pilot, main generation for WO-001."""
    import importlib.util
    import sys
    from pathlib import Path

    results_dir = Path(f"{VOLUME_RESULTS}/WO-001")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)
    ledger_path = Path(VOLUME_LEDGER)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "raw" / "gpu_log.txt"

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
        "wo_001_detection", "/root/scripts/wo_001_detection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    def _commit():
        volume.commit()

    with open(log_path, "w") as logf:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, logf)
        sys.stderr = Tee(old_err, logf)
        try:
            spec.loader.exec_module(mod)
            summary = mod.run_detection(
                model_dir=MODEL_DIR,
                revision_sha=revision_sha,
                config_path="/root/configs/wo_001.yaml",
                results_dir=str(results_dir),
                ledger_path=str(ledger_path),
                submodule_root="/root/third_party/introspection-mechanisms",
                git_hash_str=git_hash,
                gpu_type="A100-80GB",
                n_cores=GPU_CPU_CORES,
                mem_gib=GPU_MEMORY_GIB,
                volume_commit=_commit,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            _commit()

    # Pull text/binary artifacts back as base64 for large files
    import base64

    arts = {}
    for rel in [
        "generation_summary.json",
        "raw/gpu_log.txt",
        "raw/pilot_log.txt",
        "raw/inputs_snapshot.json",
        "raw/trials.jsonl",
    ]:
        p = results_dir / rel
        if p.exists():
            arts[rel] = p.read_text()
    pt = results_dir / "raw" / "concept_vectors_L37.pt"
    if pt.exists():
        arts["raw/concept_vectors_L37.pt_b64"] = base64.b64encode(pt.read_bytes()).decode(
            "ascii"
        )
    ledger_text = ledger_path.read_text() if ledger_path.exists() else ""
    return {
        "summary_keys": list((summary or {}).keys()),
        "stopped_before_main": bool((summary or {}).get("stopped_before_main")),
        "artifacts": arts,
        "budget_ledger.csv": ledger_text,
    }


@app.function(
    volumes={"/models": volume},
    secrets=[openai_secret],
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=8192,
)
def wo_001_judge() -> dict:
    """CPU: OpenAI judging of WO-001 trials."""
    import importlib.util
    import json
    import os
    from pathlib import Path

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
        return {"blocked": True, "reason": "OPENAI_API_KEY missing from openai-key secret"}

    results_dir = Path(f"{VOLUME_RESULTS}/WO-001")
    snap = json.loads((results_dir / "raw" / "inputs_snapshot.json").read_text())

    spec = importlib.util.spec_from_file_location(
        "wo_001_judge", "/root/scripts/wo_001_judge.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    summary = mod.run_judge(
        results_dir=str(results_dir),
        claims_prompt=snap["claims_detection_grading_prompt"],
        ident_prompt=snap["correct_concept_identification_grading_prompt"],
    )
    volume.commit()
    trials = (results_dir / "raw" / "trials.jsonl").read_text()
    return {
        "blocked": False,
        "judge_summary": summary,
        "trials.jsonl": trials,
        "judge_summary.json": json.dumps(summary, indent=2),
    }


@app.function(
    gpu="A100-80GB",
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=90 * 60,
    cpu=GPU_CPU_CORES,
    memory=GPU_MEMORY_MIB,
)
def wo_002_partA(git_hash: str, revision_sha: str) -> dict:
    import importlib.util
    import sys
    from pathlib import Path

    results_dir = Path(f"{VOLUME_RESULTS}/WO-002")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)
    ledger_path = Path(VOLUME_LEDGER)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "raw" / "partA_gpu_log.txt"

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
        "wo_002_partA", "/root/scripts/wo_002_partA.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    def _commit():
        volume.commit()

    with open(log_path, "w") as logf:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, logf)
        sys.stderr = Tee(old_err, logf)
        try:
            spec.loader.exec_module(mod)
            summary = mod.run_part_a(
                model_dir=MODEL_DIR,
                revision_sha=revision_sha,
                config_path="/root/configs/wo_002.yaml",
                results_dir=str(results_dir),
                ledger_path=str(ledger_path),
                submodule_root="/root/third_party/introspection-mechanisms",
                wo001_vectors_path="/root/results/WO-001/raw/concept_vectors_L37.pt",
                wo001_snapshot_path="/root/results/WO-001/raw/inputs_snapshot.json",
                wo001_per_concept_csv="/root/results/WO-001/raw/per_concept.csv",
                git_hash_str=git_hash,
                gpu_type="A100-80GB",
                n_cores=GPU_CPU_CORES,
                mem_gib=GPU_MEMORY_GIB,
                volume_commit=_commit,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            _commit()

    return {
        "l0_star": (summary or {}).get("partA", {}).get("l0_star"),
        "stopped_before_partB": bool((summary or {}).get("stopped_before_partB")),
        "projected_hours": (summary or {}).get("projected_hours"),
        "budget_ledger.csv": ledger_path.read_text() if ledger_path.exists() else "",
        "partA_summary.json": (results_dir / "partA_summary.json").read_text(),
    }


@app.function(
    gpu="A100-80GB",
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=90 * 60,
    cpu=GPU_CPU_CORES,
    memory=GPU_MEMORY_MIB,
)
def wo_002_partB(git_hash: str, revision_sha: str) -> dict:  # noqa: ARG001 — hf for datasets cache env
    import importlib.util
    import sys
    from pathlib import Path

    results_dir = Path(f"{VOLUME_RESULTS}/WO-002")
    ledger_path = Path(VOLUME_LEDGER)
    log_path = results_dir / "raw" / "partB_gpu_log.txt"

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
        "wo_002_partB", "/root/scripts/wo_002_partB.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    def _commit():
        volume.commit()

    with open(log_path, "w") as logf:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, logf)
        sys.stderr = Tee(old_err, logf)
        try:
            spec.loader.exec_module(mod)
            summary = mod.run_part_b(
                model_dir=MODEL_DIR,
                revision_sha=revision_sha,
                config_path="/root/configs/wo_002.yaml",
                results_dir=str(results_dir),
                ledger_path=str(ledger_path),
                wo001_vectors_path="/root/results/WO-001/raw/concept_vectors_L37.pt",
                wo001_snapshot_path="/root/results/WO-001/raw/inputs_snapshot.json",
                git_hash_str=git_hash,
                gpu_type="A100-80GB",
                n_cores=GPU_CPU_CORES,
                mem_gib=GPU_MEMORY_GIB,
                volume_commit=_commit,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            _commit()

    return {
        "n_errors": len((summary or {}).get("errors", [])),
        "mmlu_acc": (summary or {}).get("mmlu_acc"),
        "budget_ledger.csv": ledger_path.read_text() if ledger_path.exists() else "",
        "partB_summary.json": (results_dir / "partB_summary.json").read_text(),
    }


@app.function(
    volumes={"/models": volume},
    secrets=[openai_secret],
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=8192,
)
def wo_002_judge() -> dict:
    import importlib.util
    import json
    import os
    from pathlib import Path

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
        return {"blocked": True, "reason": "OPENAI_API_KEY missing"}

    results_dir = Path(f"{VOLUME_RESULTS}/WO-002")
    wo001_snap = json.loads(
        Path("/root/results/WO-001/raw/inputs_snapshot.json").read_text()
    )
    spec = importlib.util.spec_from_file_location(
        "wo_002_judge", "/root/scripts/wo_002_judge.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    summary = mod.run_judge(
        str(results_dir),
        wo001_snap["claims_detection_grading_prompt"],
        wo001_snap["correct_concept_identification_grading_prompt"],
    )
    volume.commit()
    return {
        "blocked": False,
        "judge_summary": summary,
        "judge_summary.json": json.dumps(summary, indent=2),
    }


def _read_revision_sha(rev_path: Path) -> str | None:
    """Parse revision_sha from model_revision.yaml without PyYAML (local entrypoint)."""
    if not rev_path.exists():
        return None
    for line in rev_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("revision_sha:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def _merge_ledger(local_ledger: Path, remote_csv: str) -> None:
    if not remote_csv:
        return
    remote_lines = remote_csv.strip().splitlines()
    if not remote_lines:
        return
    if not local_ledger.exists() or local_ledger.stat().st_size == 0:
        local_ledger.write_text(
            remote_csv if remote_csv.endswith("\n") else remote_csv + "\n"
        )
        return
    existing = local_ledger.read_text()
    with local_ledger.open("a") as f:
        for line in remote_lines[1:]:
            if line and line not in existing:
                f.write(line + "\n")


@app.local_entrypoint()
def main(action: str = "help"):
    """
    Actions:
      freeze | download | manifest | smoke | all | wo001 | wo001_judge
    """
    root = Path(__file__).resolve().parent
    results = root / "results" / "WO-000"
    results.mkdir(parents=True, exist_ok=True)
    (results / "raw").mkdir(parents=True, exist_ok=True)

    if action == "help":
        print(
            "Usage: modal run modal_app.py --action "
            "[freeze|download|manifest|smoke|all|wo001|wo001_judge|"
            "wo002_partA|wo002_partB|wo002_judge]"
        )
        return

    if action in ("wo002_partA", "wo002_partB", "wo002_judge"):
        sys.path.insert(0, str(root / "src"))
        rev = _read_revision_sha(root / "configs" / "model_revision.yaml")
        if not rev:
            raise SystemExit("missing configs/model_revision.yaml")
        out_dir = root / "results" / "WO-002"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw").mkdir(parents=True, exist_ok=True)

        def _pull_volume():
            # Refresh local results from Modal volume into results/WO-002
            # (get remote dir into parent so we don't nest WO-002/WO-002).
            dest = root / "results"
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                [
                    "modal",
                    "volume",
                    "get",
                    "--force",
                    VOLUME_NAME,
                    "results/WO-002",
                    str(dest),
                ]
            )

        if action == "wo002_partA":
            from sfo.provenance import assert_clean_pushed_main

            gh = assert_clean_pushed_main(str(root))
            print(f"=== wo_002_partA git_hash={gh} ===")
            out = wo_002_partA.remote(git_hash=gh, revision_sha=rev)
            (out_dir / "partA_summary.json").write_text(out["partA_summary.json"])
            _merge_ledger(
                root / "results" / "budget_ledger.csv", out.get("budget_ledger.csv", "")
            )
            _pull_volume()
            print(f"Part A done; l0*={out.get('l0_star')}; projected={out.get('projected_hours')}")
            if out.get("stopped_before_partB"):
                print("STOPPED before Part B")
            return

        if action == "wo002_partB":
            from sfo.provenance import assert_clean_pushed_main

            gh = assert_clean_pushed_main(str(root))
            print(f"=== wo_002_partB git_hash={gh} ===")
            out = wo_002_partB.remote(git_hash=gh, revision_sha=rev)
            (out_dir / "partB_summary.json").write_text(out["partB_summary.json"])
            _merge_ledger(
                root / "results" / "budget_ledger.csv", out.get("budget_ledger.csv", "")
            )
            _pull_volume()
            print("Part B done;", out.get("mmlu_acc"))
            return

        if action == "wo002_judge":
            print("=== wo_002_judge ===")
            out = wo_002_judge.remote()
            if out.get("blocked"):
                print("JUDGE BLOCKED:", out.get("reason"))
                return
            (out_dir / "judge_summary.json").write_text(out["judge_summary.json"])
            _pull_volume()
            print("Judge complete:", out["judge_summary"])
            return

    if action in ("wo001", "wo001_judge"):
        sys.path.insert(0, str(root / "src"))
        rev = _read_revision_sha(root / "configs" / "model_revision.yaml")
        if not rev:
            raise SystemExit("missing configs/model_revision.yaml")
        out_dir = root / "results" / "WO-001"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw").mkdir(parents=True, exist_ok=True)

        if action == "wo001":
            # Provenance gate before GPU (WO-001 defect fix)
            from sfo.provenance import assert_clean_pushed_main

            gh = assert_clean_pushed_main(str(root))
            print(f"=== wo_001_detection git_hash={gh} ===")
            out = wo_001_detection.remote(git_hash=gh, revision_sha=rev)
            import base64

            for rel, text in out["artifacts"].items():
                if rel.endswith("_b64"):
                    target = out_dir / rel.replace("_b64", "")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(base64.b64decode(text))
                else:
                    target = out_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text)
            _merge_ledger(root / "results" / "budget_ledger.csv", out.get("budget_ledger.csv", ""))
            print("WO-001 generation artifacts written")
            if out.get("stopped_before_main"):
                print("STOPPED before main run (see generation_summary.json)")
            return

        if action == "wo001_judge":
            print("=== wo_001_judge ===")
            out = wo_001_judge.remote()
            if out.get("blocked"):
                (out_dir / "JUDGE_BLOCKED.txt").write_text(out.get("reason", "blocked"))
                print("JUDGE BLOCKED:", out.get("reason"))
                return
            (out_dir / "raw" / "trials.jsonl").write_text(out["trials.jsonl"])
            (out_dir / "judge_summary.json").write_text(out["judge_summary.json"])
            print("Judge complete:", out["judge_summary"])
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
