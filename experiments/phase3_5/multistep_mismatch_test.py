#!/usr/bin/env python3
"""One-shot synthetic test access for frozen Phase 3.5-MS2 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.multistep_mismatch import (  # noqa: E402
    _assert_clean_source_tree,
    _build_configs,
    _canonical,
    _select,
    expand_runs,
    load_matrix,
)
from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.operators import build_response_operator  # noqa: E402
from src.phase35.multistep.synthetic import SyntheticSpec, generate_synthetic_split  # noqa: E402
from src.phase35.multistep.training import (  # noqa: E402
    evaluate_operator,
    structural_diagnostics,
)


FROZEN_EXECUTION_PATHS = (
    "configs/phase3_5/multistep_mismatch_matrix.json",
    "experiments/phase3_5/multistep_mismatch.py",
    "src/phase35/multistep/contracts.py",
    "src/phase35/multistep/operators.py",
    "src/phase35/multistep/synthetic.py",
    "src/phase35/multistep/training.py",
)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _assert_frozen_code_equivalent(training_sha: str) -> None:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{training_sha}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"training commit is unavailable locally: {training_sha}")
    compared = subprocess.run(
        ["git", "diff", "--quiet", training_sha, "HEAD", "--", *FROZEN_EXECUTION_PATHS],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(
            f"frozen MS2 execution code differs from training commit {training_sha}"
        )
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking frozen MS2 code equivalence")


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


def _episode_metrics(batch, prediction: torch.Tensor) -> dict:
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


def _validate_frozen_run(
    output_dir: Path,
    matrix: dict,
    candidate_id: str,
    seed: int,
    operator_config,
    training_config,
    synthetic_spec,
) -> tuple[dict, Path]:
    manifest_path = output_dir / "manifest.json"
    checkpoint_path = output_dir / "checkpoint_best_val.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen MS2 manifest/checkpoint missing: {output_dir}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_spec = replace(
        synthetic_spec, seed=synthetic_spec.seed + seed * 1_000_003
    )
    expected = {
        "protocol_version": matrix["protocol_version"],
        "route_id": candidate_id,
        "seed": seed,
        "operator_config": operator_config.to_dict(),
        "training_config": asdict(training_config),
        "synthetic_spec": asdict(expected_spec),
        "test_accessed": False,
    }
    mismatches = [
        key for key, value in expected.items()
        if _canonical(manifest.get(key)) != _canonical(value)
    ]
    if manifest.get("checkpoint_sha256") != _sha256(checkpoint_path):
        mismatches.append("checkpoint_sha256")
    if mismatches:
        raise RuntimeError(
            f"frozen MS2 run mismatch for {candidate_id}/seed={seed}: "
            f"{sorted(set(mismatches))}"
        )
    _assert_frozen_code_equivalent(manifest["git_sha"])
    return manifest, checkpoint_path


def evaluate_one(
    output_dir: Path,
    matrix: dict,
    regime: dict,
    candidate: dict,
    seed: int,
    device: str,
    smoke: bool,
) -> dict:
    candidate_id = candidate["candidate_id"]
    metrics_path = output_dir / "metrics_test.json"
    episode_path = output_dir / "episode_metrics_test.json"
    ledger_path = output_dir / "synthetic_test_access_ledger.json"
    if metrics_path.exists() or episode_path.exists() or ledger_path.exists():
        raise RuntimeError(f"refusing repeat or partial MS2 synthetic test access: {output_dir}")
    operator_config, training_config, synthetic_spec, _ = _build_configs(
        matrix, regime, candidate, smoke
    )
    manifest, checkpoint_path = _validate_frozen_run(
        output_dir,
        matrix,
        candidate_id,
        seed,
        operator_config,
        training_config,
        synthetic_spec,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("protocol_version", "route_id", "seed", "git_sha"):
        if checkpoint.get(key) != manifest.get(key):
            raise RuntimeError(f"checkpoint/manifest mismatch for {key}: {output_dir}")
    test_samples = 32 if smoke else int(matrix["synthetic_defaults"]["test_samples"])
    ledger = {
        "protocol_version": matrix["protocol_version"],
        "status": "started",
        "evidence_scope": "synthetic_mismatch_test_not_field_causality",
        "candidate_id": candidate_id,
        "regime_id": regime["regime_id"],
        "seed": seed,
        "training_git_sha": manifest["git_sha"],
        "evaluation_git_sha": _git_sha(),
        "frozen_code_equivalence_paths": list(FROZEN_EXECUTION_PATHS),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_selector": manifest["checkpoint_selector"],
        "test_samples": test_samples,
    }
    _json_dump(ledger_path, ledger)
    operator = build_response_operator(OperatorConfig.from_mapping(checkpoint["operator_config"]))
    operator.load_state_dict(checkpoint["model_state_dict"])
    dev = torch.device(device)
    operator.to(dev)
    checkpoint_spec = SyntheticSpec(**checkpoint["synthetic_spec"])
    test_batch = generate_synthetic_split(
        replace(checkpoint_spec, samples=test_samples), "test"
    )
    metrics, prediction = evaluate_operator(operator, test_batch, dev)
    metrics["structural_diagnostics"] = structural_diagnostics(operator, test_batch, dev)
    metrics["truth"] = test_batch.truth
    episodes = _episode_metrics(test_batch, prediction)
    _json_dump(metrics_path, metrics)
    _json_dump(episode_path, episodes)
    ledger["status"] = "completed"
    ledger["trajectory_design_sha256"] = episodes["trajectory_design_sha256"]
    _json_dump(ledger_path, ledger)
    manifest["test_accessed"] = True
    manifest["test_access_note"] = "synthetic_mismatch_known_truth_only"
    manifest["test_access_ledger"] = ledger_path.name
    manifest["test_episode_metrics"] = episode_path.name
    _json_dump(output_dir / "manifest.json", manifest)
    return {
        "regime_id": regime["regime_id"],
        "candidate_id": candidate_id,
        "seed": seed,
        "status": "synthetic_test_completed",
        "effect_mae": metrics["effect_mae"],
        "clean_effect_nmae": metrics["clean_effect_nmae"],
        "trajectory_design_sha256": episodes["trajectory_design_sha256"],
    }


def _completed_run(output_dir: Path) -> bool:
    required = [
        output_dir / "manifest.json",
        output_dir / "metrics_test.json",
        output_dir / "episode_metrics_test.json",
        output_dir / "synthetic_test_access_ledger.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    ledger = json.loads(required[3].read_text(encoding="utf-8"))
    if manifest.get("test_accessed") is not True or ledger.get("status") != "completed":
        raise RuntimeError(f"inconsistent completed MS2 test artifacts: {output_dir}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"),
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/multistep_mismatch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--evaluate-test-matrix", action="store_true")
    parser.add_argument("--allow-synthetic-test", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_synthetic_test:
        raise SystemExit("MS2 test requires explicit --allow-synthetic-test")
    if args.evaluate_test == args.evaluate_test_matrix:
        raise SystemExit("choose exactly one of --evaluate-test or --evaluate-test-matrix")
    matrix_path = Path(args.matrix).resolve()
    canonical_matrix_path = (
        ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"
    ).resolve()
    if not args.smoke and matrix_path != canonical_matrix_path:
        raise SystemExit(
            "formal MS2 test requires the repository-frozen mismatch matrix"
        )
    matrix = load_matrix(matrix_path)
    allowed_seeds = {int(seed) for seed in matrix["seeds"]}
    if args.evaluate_test:
        if args.candidate_id is None or args.seed is None:
            raise SystemExit("--evaluate-test requires --candidate-id and --seed")
        selected = [{"candidate_id": args.candidate_id, "seed": args.seed}]
    else:
        if args.candidate_id is not None or args.seed is not None or args.smoke:
            raise SystemExit("matrix test does not accept candidate/seed/smoke filters")
        selected = expand_runs(matrix)
    output_root = Path(args.output_root).resolve()
    if not args.smoke:
        _assert_clean_source_tree(output_root, allow_generated_outputs=args.skip_existing)
    completed = []
    for index, run in enumerate(selected, start=1):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        if seed not in allowed_seeds:
            raise SystemExit(f"seed={seed} is not frozen in matrix")
        regime, candidate = _select(matrix, candidate_id)
        output_dir = output_root / f"ms2_{candidate_id}_s{seed}"
        if args.skip_existing and _completed_run(output_dir):
            completed.append({
                "regime_id": regime["regime_id"],
                "candidate_id": candidate_id,
                "seed": seed,
                "status": "skipped_completed_test",
            })
            continue
        print(
            f"[{index}/{len(selected)}] test regime={regime['regime_id']} "
            f"candidate={candidate_id} seed={seed}",
            file=sys.stderr,
            flush=True,
        )
        completed.append(evaluate_one(
            output_dir, matrix, regime, candidate, seed, args.device, args.smoke
        ))
    payload = completed[0] if len(completed) == 1 else {
        "status": "synthetic_test_matrix_completed",
        "run_count": len(completed),
        "runs": completed,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
