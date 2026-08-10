#!/usr/bin/env python3
"""One-shot synthetic test access for frozen Phase 3.5-MS2-J checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.joint_coupling import (  # noqa: E402
    FROZEN_EXECUTION_PATHS,
    _build_configs,
    _select,
    expand_runs,
    load_matrix,
)
from experiments.phase3_5.multistep_mismatch import (  # noqa: E402
    _assert_clean_source_tree,
    _canonical,
)
from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.operators import build_response_operator  # noqa: E402
from src.phase35.multistep.synthetic import (  # noqa: E402
    SyntheticSpec,
    generate_synthetic_split,
)
from src.phase35.multistep.staging import environment_payload  # noqa: E402
from src.phase35.multistep.training import (  # noqa: E402
    evaluate_operator,
    structural_diagnostics,
)


TEST_PROTOCOL_VERSION = "phase3.5-ms2j-test-v1"
DEFAULT_AUTHORIZATION = (
    ROOT / "configs/phase3_5/joint_coupling_test_authorization.json"
)
ROOT_LEDGER_NAME = "synthetic_test_matrix_access_ledger.json"
RUN_LEDGER_NAME = "synthetic_test_access_ledger.json"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required MS2-J test artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _resolve_repo_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"authorization path must be repository-relative: {relative}")
    resolved = (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"authorization path escapes repository: {relative}")
    return resolved


def load_authorization(path: str | Path) -> dict[str, Any]:
    authorization = _read_json(Path(path))
    required = {
        "protocol_version",
        "training_protocol_version",
        "evidence_scope",
        "matrix",
        "validation_summary",
        "checkpoint_archive",
        "expected_run_count",
        "test_samples",
        "bootstrap",
        "gates",
        "frozen_validation_status",
        "authorized_scope",
    }
    missing = sorted(required - set(authorization))
    if missing:
        raise ValueError(f"MS2-J test authorization missing keys: {missing}")
    if authorization["protocol_version"] != TEST_PROTOCOL_VERSION:
        raise ValueError(
            f"MS2-J test runner accepts only {TEST_PROTOCOL_VERSION}"
        )
    scope = authorization["authorized_scope"]
    expected_scope = {
        "synthetic_test_once": True,
        "retraining": False,
        "hyperparameter_changes": False,
        "additional_seeds": False,
        "real_data_test": False,
    }
    if scope != expected_scope:
        raise ValueError("MS2-J test authorization scope is not fail-closed")
    for key in ("matrix", "validation_summary", "checkpoint_archive"):
        item = authorization[key]
        if not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
            raise ValueError(f"invalid pinned sha256 for {key}")
        _resolve_repo_path(item["path"])
    if int(authorization["expected_run_count"]) != 27:
        raise ValueError("MS2-J test authorization must freeze 27 runs")
    if int(authorization["test_samples"]) < 1:
        raise ValueError("MS2-J test_samples must be positive")
    if int(authorization["bootstrap"]["replicates"]) < 1000:
        raise ValueError("MS2-J formal bootstrap requires at least 1000 replicates")
    return authorization


def _assert_pinned(path: Path, expected_sha256: str, label: str) -> None:
    actual = _sha256(path)
    if actual == expected_sha256:
        return
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        relative = ""
    if relative:
        clean = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=ROOT,
            check=False,
        )
        try:
            blob = subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            blob = b""
        blob_sha256 = _sha256_bytes(blob) if blob else None
        if clean.returncode == 0 and blob_sha256 == expected_sha256:
            return
    raise RuntimeError(
        f"pinned {label} sha256 mismatch: expected={expected_sha256}, "
        f"working_tree={actual}"
    )


def _assert_frozen_code_equivalent(training_sha: str) -> None:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{training_sha}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"MS2-J training commit unavailable: {training_sha}")
    compared = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            training_sha,
            "HEAD",
            "--",
            *FROZEN_EXECUTION_PATHS,
        ],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(
            f"frozen MS2-J execution code differs from {training_sha}"
        )
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking frozen MS2-J code equivalence")


def _trajectory_digest(batch) -> str:
    digest = hashlib.sha256()
    for tensor in (
        batch.context,
        batch.action,
        batch.reference,
        batch.clean_effect,
        batch.profile_ids,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _episode_metrics(batch, prediction: torch.Tensor) -> dict[str, Any]:
    prediction = prediction.detach().cpu()
    target = batch.target_effect.detach().cpu()
    clean = batch.clean_effect.detach().cpu()
    observed_error = (prediction - target).abs()
    clean_error = (prediction - clean).abs()
    horizon = clean.shape[1]
    points = sorted(set(min(point, horizon) for point in (1, 6, 18, 60)))
    return {
        "episode_ids": list(range(clean.shape[0])),
        "profile_ids": batch.profile_ids.detach().cpu().tolist(),
        "profile_names": list(batch.profile_names),
        "trajectory_design_sha256": _trajectory_digest(batch),
        "observed_effect_mae": observed_error.mean(dim=1).tolist(),
        "clean_effect_mae": clean_error.mean(dim=1).tolist(),
        "clean_effect_scale": clean.abs().mean(dim=1).tolist(),
        "clean_horizon_absolute_error": {
            f"H{point}": clean_error[:, point - 1].tolist() for point in points
        },
    }


def _member_bytes(
    archive: tarfile.TarFile,
    member_name: str,
    expected_sha256: str,
) -> bytes:
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"required checkpoint archive member missing: {member_name}"
        ) from exc
    if not member.isfile():
        raise RuntimeError(f"checkpoint archive member is not a file: {member_name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"cannot read checkpoint archive member: {member_name}")
    value = handle.read()
    actual_sha256 = _sha256_bytes(value)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checkpoint archive member hash mismatch: {member_name}"
        )
    return value


def _unauthorized_artifacts(run_dir: Path) -> list[str]:
    names = (
        "metrics_test.json",
        "episode_metrics_test.json",
        "metrics_stage_a_test.json",
        "episode_metrics_stage_a_test.json",
        RUN_LEDGER_NAME,
    )
    return [name for name in names if (run_dir / name).exists()]


def preflight(
    authorization_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[dict[str, Any]]]:
    authorization_path = Path(authorization_path).resolve()
    authorization = load_authorization(authorization_path)
    matrix_path = _resolve_repo_path(authorization["matrix"]["path"])
    validation_path = _resolve_repo_path(
        authorization["validation_summary"]["path"]
    )
    archive_path = _resolve_repo_path(
        authorization["checkpoint_archive"]["path"]
    )
    _assert_pinned(matrix_path, authorization["matrix"]["sha256"], "matrix")
    _assert_pinned(
        validation_path,
        authorization["validation_summary"]["sha256"],
        "validation summary",
    )
    _assert_pinned(
        archive_path,
        authorization["checkpoint_archive"]["sha256"],
        "checkpoint archive",
    )
    matrix = load_matrix(matrix_path)
    if matrix["protocol_version"] != authorization["training_protocol_version"]:
        raise RuntimeError("training matrix protocol does not match test authorization")
    validation = _read_json(validation_path)
    observed_validation_status = {
        "all_gates_pass": validation.get("all_gates_pass"),
        "joint_module_gate_pass": validation.get("joint_module_gate", {}).get(
            "all_seed_improvements_exceed_20pct"
        ),
        "staged_stability_gate_pass": validation.get(
            "staged_stability_gate", {}
        ).get("all_seeds_pass"),
    }
    if observed_validation_status != authorization["frozen_validation_status"]:
        raise RuntimeError("validation status differs from frozen test authorization")
    if validation.get("test_accessed") is not False:
        raise RuntimeError("validation summary reports prior test access")
    runs = expand_runs(matrix)
    if len(runs) != int(authorization["expected_run_count"]):
        raise RuntimeError("expanded MS2-J run count differs from authorization")
    output_root = validation_path.parent
    root_outputs = [ROOT_LEDGER_NAME, "summary_test.json"]
    existing_root = [name for name in root_outputs if (output_root / name).exists()]
    if existing_root:
        raise RuntimeError(
            f"refusing repeat or partial MS2-J matrix test access: {existing_root}"
        )
    records: list[dict[str, Any]] = []
    execution_shas: set[str] = set()
    with tarfile.open(archive_path, "r") as archive:
        file_members = [member for member in archive.getmembers() if member.isfile()]
        if len(file_members) != int(
            authorization["checkpoint_archive"]["member_count"]
        ):
            raise RuntimeError("checkpoint archive member count mismatch")
        for member in file_members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe checkpoint archive member: {member.name}")
        for run in runs:
            candidate_id, seed = run["candidate_id"], int(run["seed"])
            regime, candidate = _select(matrix, candidate_id)
            operator, training, staged, synthetic, _ = _build_configs(
                matrix, regime, candidate, False
            )
            run_dir = output_root / f"ms2j_{candidate_id}_s{seed}"
            manifest = _read_json(run_dir / "manifest.json")
            unauthorized = _unauthorized_artifacts(run_dir)
            if unauthorized:
                raise RuntimeError(
                    f"refusing repeat or partial MS2-J run test access: "
                    f"{candidate_id}/seed={seed}: {unauthorized}"
                )
            expected_spec = replace(
                synthetic, seed=synthetic.seed + seed * 1_000_003
            )
            expected = {
                "protocol_version": matrix["protocol_version"],
                "route_id": candidate_id,
                "seed": seed,
                "training_mode": candidate["training_mode"],
                "operator_config": operator.to_dict(),
                "training_config": asdict(training),
                "synthetic_spec": asdict(expected_spec),
                "test_accessed": False,
            }
            if candidate["training_mode"] == "staged":
                expected["staged_training_config"] = asdict(staged)
            mismatches = [
                key
                for key, expected_value in expected.items()
                if _canonical(manifest.get(key)) != _canonical(expected_value)
            ]
            if mismatches:
                raise RuntimeError(
                    f"frozen MS2-J run mismatch for {candidate_id}/seed={seed}: "
                    f"{mismatches}"
                )
            execution_sha = manifest.get("git_sha")
            if not execution_sha:
                raise RuntimeError(f"missing training git SHA: {candidate_id}/seed={seed}")
            execution_shas.add(execution_sha)
            member_name = f"{run_dir.name}/checkpoint_best_val.pt"
            checkpoint_bytes = _member_bytes(
                archive, member_name, manifest["checkpoint_sha256"]
            )
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
            )
            for key in ("protocol_version", "route_id", "seed", "git_sha"):
                if checkpoint.get(key) != manifest.get(key):
                    raise RuntimeError(
                        f"checkpoint/manifest mismatch for {key}: "
                        f"{candidate_id}/seed={seed}"
                    )
            if any(
                torch.is_tensor(value) and not torch.isfinite(value).all()
                for value in checkpoint.get("model_state_dict", {}).values()
            ):
                raise RuntimeError(
                    f"non-finite canonical checkpoint: {candidate_id}/seed={seed}"
                )
            stage_a = None
            if candidate["training_mode"] == "staged":
                stage_info = manifest.get("stage_checkpoints", {}).get("stage_a", {})
                if stage_info.get("file") != "checkpoint_stage_a.pt":
                    raise RuntimeError("staged run has no canonical Stage-A checkpoint")
                stage_member = f"{run_dir.name}/{stage_info['file']}"
                stage_bytes = _member_bytes(
                    archive, stage_member, stage_info["sha256"]
                )
                stage_checkpoint = torch.load(
                    io.BytesIO(stage_bytes), map_location="cpu", weights_only=False
                )
                if stage_checkpoint.get("stage") != "stage_a":
                    raise RuntimeError("Stage-A archive payload has the wrong stage")
                if any(
                    torch.is_tensor(value) and not torch.isfinite(value).all()
                    for value in stage_checkpoint.get("model_state_dict", {}).values()
                ):
                    raise RuntimeError(
                        f"non-finite Stage-A checkpoint: {candidate_id}/seed={seed}"
                    )
                stage_a = {
                    "member": stage_member,
                    "sha256": stage_info["sha256"],
                }
            records.append(
                {
                    **run,
                    "run_dir": run_dir,
                    "manifest": manifest,
                    "checkpoint_member": member_name,
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "stage_a": stage_a,
                }
            )
    if len(execution_shas) != 1:
        raise RuntimeError(f"MS2-J runs use multiple execution SHAs: {execution_shas}")
    _assert_frozen_code_equivalent(next(iter(execution_shas)))
    return authorization, matrix, output_root, archive_path, records


def _evaluate_checkpoint_bytes(
    checkpoint_bytes: bytes,
    test_samples: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dev = torch.device(device)
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location=dev, weights_only=False
    )
    operator = build_response_operator(
        OperatorConfig.from_mapping(checkpoint["operator_config"])
    ).to(dev)
    operator.load_state_dict(checkpoint["model_state_dict"])
    operator.eval()
    spec = SyntheticSpec(**checkpoint["synthetic_spec"])
    batch = generate_synthetic_split(replace(spec, samples=test_samples), "test")
    metrics, prediction = evaluate_operator(operator, batch, dev)
    metrics["structural_diagnostics"] = structural_diagnostics(
        operator, batch, dev
    )
    metrics["truth"] = batch.truth
    return metrics, _episode_metrics(batch, prediction)


def execute_test_matrix(
    authorization_path: str | Path,
    device: str,
) -> dict[str, Any]:
    authorization, _, output_root, archive_path, records = preflight(
        authorization_path
    )
    _assert_clean_source_tree(output_root, allow_generated_outputs=True)
    root_ledger_path = output_root / ROOT_LEDGER_NAME
    root_ledger = {
        "protocol_version": authorization["protocol_version"],
        "status": "started",
        "evidence_scope": authorization["evidence_scope"],
        "authorization": str(Path(authorization_path).resolve().relative_to(ROOT)),
        "authorization_sha256": _sha256(Path(authorization_path).resolve()),
        "matrix_sha256": authorization["matrix"]["sha256"],
        "validation_summary_sha256": authorization["validation_summary"]["sha256"],
        "checkpoint_archive_sha256": authorization["checkpoint_archive"]["sha256"],
        "evaluation_git_sha": _git_sha(),
        "environment": environment_payload(torch.device(device)),
        "run_count": len(records),
        "test_samples": int(authorization["test_samples"]),
        "completed_runs": [],
    }
    _json_dump(root_ledger_path, root_ledger)
    completed = []
    with tarfile.open(archive_path, "r") as archive:
        for index, record in enumerate(records, start=1):
            run_dir: Path = record["run_dir"]
            manifest = record["manifest"]
            candidate_id, seed = record["candidate_id"], int(record["seed"])
            print(
                f"[{index}/{len(records)}] MS2-J test candidate={candidate_id} "
                f"seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            ledger_path = run_dir / RUN_LEDGER_NAME
            ledger = {
                "protocol_version": authorization["protocol_version"],
                "status": "started",
                "evidence_scope": authorization["evidence_scope"],
                "candidate_id": candidate_id,
                "regime_id": record["regime_id"],
                "seed": seed,
                "training_git_sha": manifest["git_sha"],
                "evaluation_git_sha": _git_sha(),
                "checkpoint_archive": str(archive_path.relative_to(ROOT)),
                "checkpoint_archive_sha256": authorization[
                    "checkpoint_archive"
                ]["sha256"],
                "checkpoint_member": record["checkpoint_member"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "checkpoint_selector": manifest["checkpoint_selector"],
                "test_samples": int(authorization["test_samples"]),
            }
            if record["stage_a"] is not None:
                ledger["stage_a_checkpoint_member"] = record["stage_a"]["member"]
                ledger["stage_a_checkpoint_sha256"] = record["stage_a"]["sha256"]
            _json_dump(ledger_path, ledger)
            checkpoint_bytes = _member_bytes(
                archive,
                record["checkpoint_member"],
                record["checkpoint_sha256"],
            )
            metrics, episodes = _evaluate_checkpoint_bytes(
                checkpoint_bytes,
                int(authorization["test_samples"]),
                device,
            )
            _json_dump(run_dir / "metrics_test.json", metrics)
            _json_dump(run_dir / "episode_metrics_test.json", episodes)
            if record["stage_a"] is not None:
                stage_bytes = _member_bytes(
                    archive,
                    record["stage_a"]["member"],
                    record["stage_a"]["sha256"],
                )
                stage_metrics, stage_episodes = _evaluate_checkpoint_bytes(
                    stage_bytes,
                    int(authorization["test_samples"]),
                    device,
                )
                if (
                    stage_episodes["trajectory_design_sha256"]
                    != episodes["trajectory_design_sha256"]
                ):
                    raise RuntimeError("Stage-A and final test trajectories are unpaired")
                _json_dump(run_dir / "metrics_stage_a_test.json", stage_metrics)
                _json_dump(
                    run_dir / "episode_metrics_stage_a_test.json", stage_episodes
                )
            ledger["status"] = "completed"
            ledger["trajectory_design_sha256"] = episodes[
                "trajectory_design_sha256"
            ]
            _json_dump(ledger_path, ledger)
            manifest["test_accessed"] = True
            manifest["test_access_note"] = (
                "synthetic_joint_coupling_known_truth_only"
            )
            manifest["test_access_ledger"] = RUN_LEDGER_NAME
            manifest["test_episode_metrics"] = "episode_metrics_test.json"
            if record["stage_a"] is not None:
                manifest["test_stage_a_episode_metrics"] = (
                    "episode_metrics_stage_a_test.json"
                )
            _json_dump(run_dir / "manifest.json", manifest)
            item = {
                "candidate_id": candidate_id,
                "seed": seed,
                "status": "synthetic_test_completed",
                "clean_effect_nmae": metrics["clean_effect_nmae"],
                "trajectory_design_sha256": episodes[
                    "trajectory_design_sha256"
                ],
            }
            completed.append(item)
            root_ledger["completed_runs"].append(
                {"candidate_id": candidate_id, "seed": seed}
            )
            _json_dump(root_ledger_path, root_ledger)
    root_ledger["status"] = "completed"
    _json_dump(root_ledger_path, root_ledger)
    return {
        "protocol_version": authorization["protocol_version"],
        "status": "synthetic_test_matrix_completed",
        "run_count": len(completed),
        "runs": completed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-test-matrix", action="store_true")
    parser.add_argument("--allow-synthetic-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization_path = Path(args.authorization).resolve()
    if args.dry_run:
        if args.evaluate_test_matrix or args.allow_synthetic_test:
            raise SystemExit("--dry-run cannot be combined with test access flags")
        authorization, _, _, _, records = preflight(authorization_path)
        print(
            json.dumps(
                {
                    "protocol_version": authorization["protocol_version"],
                    "evidence_scope": authorization["evidence_scope"],
                    "run_count": len(records),
                    "archive_member_count": authorization[
                        "checkpoint_archive"
                    ]["member_count"],
                    "frozen_validation_status": authorization[
                        "frozen_validation_status"
                    ],
                    "test_accessed": False,
                },
                indent=2,
            )
        )
        return
    if not args.evaluate_test_matrix:
        raise SystemExit("MS2-J test requires --evaluate-test-matrix")
    if not args.allow_synthetic_test:
        raise SystemExit("MS2-J test requires explicit --allow-synthetic-test")
    if authorization_path != DEFAULT_AUTHORIZATION.resolve():
        raise SystemExit("formal MS2-J test requires the repository authorization")
    print(
        json.dumps(
            execute_test_matrix(authorization_path, args.device), indent=2
        )
    )


if __name__ == "__main__":
    main()
