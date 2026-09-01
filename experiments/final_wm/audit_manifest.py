"""Content-addressed manifest for authoritative Final-WM v0.7 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.final_wm import matrix_spec as ms
from src.final_wm.contracts import FinalWMProtocolError


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FinalWMProtocolError(f"manifest input missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception as exc:
        raise FinalWMProtocolError("cannot resolve git commit") from exc


def git_is_clean() -> bool:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
        ).stdout
    except Exception as exc:
        raise FinalWMProtocolError("cannot inspect git worktree") from exc
    return not status.strip()


def ensure_authoritative_preflight(out: str | Path) -> dict:
    """Record a clean immutable starting point before result files exist."""
    out = Path(out)
    marker = out / ".authoritative_preflight.json"
    commit = git_commit()
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("commit") != commit or payload.get("matrix_version") != ms.MATRIX_VERSION:
            raise FinalWMProtocolError("authoritative preflight does not match current commit/protocol")
        return payload
    if not git_is_clean():
        raise FinalWMProtocolError("authoritative run requires a clean worktree")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "commit": commit,
        "matrix_version": ms.MATRIX_VERSION,
        "clean_at_start": True,
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def expected_run_ids() -> list[str]:
    specs = [ms._base("dsyn", "student", seed) for seed in ms.SEEDS]
    specs += ms.o1_specs() + ms.t1_specs() + ms.b1_specs() + ms.j1_specs()
    specs += [
        ms.j1_staged_boundary_spec(seed, f"j1_staged_main_seed{seed}.pt")
        for seed in ms.SEEDS
    ]
    return sorted(f"{spec.unit}_{spec.arm}_seed{spec.seed}" for spec in specs)


def _all_units_executed(summary: dict) -> bool:
    units = summary.get("units", {})
    if set(units) < {"o1", "t1", "b1", "j1", "r1"}:
        return False
    for name in ("o1", "t1"):
        block = units[name]
        if not isinstance(block, dict) or not block:
            return False
        if any(item.get("status") not in ("COMPLETE", "INCOMPLETE") for item in block.values()):
            return False
    return all(
        units[name].get("status") in ("COMPLETE", "INCOMPLETE")
        for name in ("b1", "j1", "r1")
    )


def create_authoritative_manifest(
    *,
    out: str | Path,
    summary_path: str | Path,
    record_path: str | Path,
    properties_path: str | Path,
    side: str,
    command: list[str] | None = None,
) -> Path:
    out = Path(out)
    preflight = ensure_authoritative_preflight(out)
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("quick") or summary.get("matrix_version") != ms.MATRIX_VERSION:
        raise FinalWMProtocolError("authoritative manifest requires a full v0.7 summary")
    if summary.get("side") != side or side not in ("A", "B"):
        raise FinalWMProtocolError("authoritative manifest side mismatch")
    if not _all_units_executed(summary):
        raise FinalWMProtocolError("authoritative manifest requires every v0.7 unit to execute")

    dsyn_path = out / "dsyn_verdict.json"
    dsyn = json.loads(dsyn_path.read_text(encoding="utf-8"))
    if dsyn.get("quick") or dsyn.get("verdict") != "PASS":
        raise FinalWMProtocolError("authoritative manifest requires full D-SYN PASS")
    if sorted(item.get("seed") for item in dsyn.get("per_seed", [])) != list(ms.SEEDS):
        raise FinalWMProtocolError("authoritative manifest requires complete D-SYN seeds")
    if any(item.get("n_perturbed", 0) <= 0 or item.get("parameter_delta_l2", 0.0) <= 0.0
           for item in dsyn["per_seed"]):
        raise FinalWMProtocolError("authoritative manifest found a D-SYN no-op perturbation")

    expected = expected_run_ids()
    ledger_path = out / "ledger.jsonl"
    finals = {
        item["run_id"]
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if (item := json.loads(line)).get("final")
    }
    missing_ledger = sorted(set(expected) - finals)
    if missing_ledger:
        raise FinalWMProtocolError(f"authoritative ledger missing final runs: {missing_ledger}")

    required_files = [summary_path, dsyn_path, ledger_path]
    required_files += [out / "checkpoints" / f"{run_id}.pt" for run_id in expected]
    required_files += [
        out / "metrics" / f"{run_id}.pt"
        for run_id in expected
        if not run_id.startswith("j1_staged_boundary_from_")
        and not run_id.startswith("dsyn_student_")
    ]
    hashes = {
        str(path.relative_to(out)).replace("\\", "/"): sha256_file(path)
        for path in required_files
    }
    record_path = Path(record_path).resolve()
    properties_path = Path(properties_path).resolve()

    def input_entry(input_path: Path) -> dict:
        return {
            "path": str(input_path),
            "package_relative_path": os.path.relpath(input_path, out.resolve()).replace("\\", "/"),
            "sha256": sha256_file(input_path),
        }

    payload = {
        "schema_version": 1,
        "authoritative": True,
        "matrix_version": ms.MATRIX_VERSION,
        "protocol_base": "v0.6-canonical-v2.2-training-120-20",
        "side": side,
        "commit": preflight["commit"],
        "command": command or list(sys.argv),
        "seeds": list(ms.SEEDS),
        "r1_arm": ms.R1_ARM,
        "test_locked": True,
        "inputs": {
            "record": input_entry(record_path),
            "properties": input_entry(properties_path),
        },
        "expected_run_ids": expected,
        "files": hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def verify_manifest(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("authoritative") or payload.get("matrix_version") != ms.MATRIX_VERSION:
        raise FinalWMProtocolError("not an authoritative v0.7 manifest")
    if payload.get("seeds") != list(ms.SEEDS) or payload.get("r1_arm") != ms.R1_ARM:
        raise FinalWMProtocolError("manifest protocol identity mismatch")
    if payload.get("test_locked") is not True:
        raise FinalWMProtocolError("manifest must keep test locked")
    if sorted(payload.get("expected_run_ids", [])) != expected_run_ids():
        raise FinalWMProtocolError("manifest run set is incomplete or drifted")
    for item in payload.get("inputs", {}).values():
        packaged = item.get("package_relative_path")
        packaged_path = path.parent / packaged if packaged else None
        target = packaged_path if packaged_path is not None and packaged_path.is_file() else Path(item["path"])
        if sha256_file(target) != item["sha256"]:
            raise FinalWMProtocolError(f"manifest input hash mismatch: {target}")
    for relative, expected_hash in payload.get("files", {}).items():
        target = path.parent / relative
        if sha256_file(target) != expected_hash:
            raise FinalWMProtocolError(f"manifest artifact hash mismatch: {relative}")
    return {"verified": True, "manifest": str(path), "n_files": len(payload["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_manifest(args.manifest), indent=2))


if __name__ == "__main__":
    main()
