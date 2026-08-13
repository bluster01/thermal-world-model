"""Cache-free Supervisor audit assembly for returned RM3-AV artifacts.

This module verifies artifacts and builds paired evidence tables.  It never assigns
scientific verdicts: Q01-Q33 remain null until a Supervisor reads the returned data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..schema import Phase35ProtocolError
from .rm3av_contracts import RM3AVRunSpec, rm3av_run_specs, validate_rm3av_matrix
from .rm3av_diagnostics import build_manual_verdict_template


PAIR_BASELINES = {
    "C03": "C25",
    "C04": "C27", "C05": "C27", "C06": "C27",
    "C07": "C26", "C08": "C26", "C09": "C26",
    "C10": "C26", "C11": "C26",
    "C12": "C27", "C13": "C27",
    "C14": "C27", "C15": "C27", "C16": "C27",
    "C17": "C26", "C18": "C26",
    "C19": "C26", "C20": "C26", "C21": "C26", "C22": "C26",
    "C23": "C26", "C24": "C26",
    "C25": "C00", "C26": "C01", "C27": "C02",
    "C28": "C25", "C29": "C26", "C30": "C27",
    "C31": "C27",
}

EXPECTED_SHARED_INITIALIZATION = {
    "C03": ("encoder", "valve_policy", "tin", "free_residual", "response", "downstream"),
    "C04": ("encoder", "valve_policy", "tin", "response", "downstream", "bypass"),
    "C05": ("encoder", "valve_policy", "tin", "response", "downstream", "bypass"),
    "C06": ("encoder", "valve_policy", "tin", "downstream", "bypass"),
    "C07": ("encoder", "valve_policy", "tin", "response", "downstream"),
    "C08": ("encoder", "valve_policy", "tin", "response", "downstream"),
    "C09": ("encoder", "valve_policy", "tin", "free_residual", "response", "downstream"),
    "C10": ("encoder", "valve_policy", "tin", "free_residual", "response", "downstream"),
    "C11": ("encoder", "valve_policy", "tin", "free_residual", "response", "downstream"),
    "C12": ("encoder", "valve_policy", "tin", "response", "downstream", "bypass"),
    "C13": ("encoder", "valve_policy", "tin", "response", "downstream", "bypass"),
    "C14": ("encoder", "valve_policy", "tin", "response", "downstream", "bypass"),
    "C15": ("encoder", "tin", "response", "downstream", "bypass"),
    "C16": ("encoder", "tin", "response", "downstream", "bypass"),
    "C17": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C18": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C19": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C20": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C21": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C22": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C23": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C24": ("encoder", "valve_policy", "tin", "free_residual", "downstream"),
    "C28": ("encoder", "valve_policy", "tin", "free_residual", "response", "downstream"),
    "C29": ("encoder", "valve_policy", "tin", "free_residual", "response", "downstream"),
    "C30": ("encoder", "valve_policy", "tin", "response", "downstream", "bypass"),
    "C31": ("encoder", "valve_policy", "tin", "response", "downstream", "bypass"),
}

QUESTION_EVIDENCE = {
    "Q01": ["C10/C11/C12/C13 paired contrasts", "training_graph"],
    "Q02": ["C14/C15/C16 paired contrasts", "valve_trajectory", "horizon_curve_steps"],
    "Q03": ["AV0 functional_replay", "diagnostic_modes"],
    "Q04": ["C03-C09 paired contrasts", "bypass/response modes"],
    "Q05": ["C04/C05/C06", "finite_difference_response", "response_off"],
    "Q06": ["common four-task metrics", "no composite ranking"],
    "Q07": ["checkpoint_selector", "paired fold table"],
    "Q08": ["C28/C29/C30", "convergence"],
    "Q09": ["C24", "finite_difference_response"],
    "Q10": ["C07/C08/C09", "valve_policy_probes"],
    "Q11": ["initialization_fairness", "C25/C26/C27"],
    "Q12": ["C19-C23", "response_trajectory", "fold paired table"],
    "Q13": ["AV0 calibration_corrections"],
    "Q14": ["C17/C18/C26", "daily_gain_context_activity"],
    "Q15": ["C23/C24", "action_alignment_sensitivity_seconds"],
    "Q16": ["A/B metrics", "terminal_strata"],
    "Q17": ["explicit_to_total_local_change_ratio", "finite_difference_response"],
    "Q18": ["capacity/action-shield/placebo joint evidence", "assumption_ledger"],
    "Q19": ["daily_gain_context_activity"],
    "Q20": ["residualized_valve_innovation.rank", "raw_valve_change_rank"],
    "Q21": ["C19-C23 blocked-fold contrasts", "AV0 calibration is diagnostic only"],
    "Q22": ["AV0 functional_replay", "C04/C05/C06 retrained contrasts"],
    "Q23": ["C11/C12 training_graph"],
    "Q24": ["residualized_valve_innovation.rank", "raw config flag ignored"],
    "Q25": ["valve_policy_probes", "lead/wrong/shuffle modes", "claim_boundary"],
    "Q26": ["AV0 horizon curves", "AV1 horizon curves"],
    "Q27": ["C31 two-window rollout", "state_closure"],
    "Q28": ["terminal_strata.utc_date", "measured context kept separate"],
    "Q29": ["assumption_ledger"],
    "Q30": ["action_alignment_sensitivity_seconds", "lead mode"],
    "Q31": ["state_closure", "C31"],
    "Q32": ["residualized_valve_innovation.dependence", "mechanism_prediction_residual_dependence"],
    "Q33": ["state_closure", "C31"],
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_root(output_root: Path, required: set[str]) -> None:
    ledger_path = output_root / "artifact_ledger.json"
    if not ledger_path.is_file():
        raise Phase35ProtocolError("RM3-AV root artifact ledger is missing")
    ledger = _read(ledger_path)
    if set(ledger) != required - {"artifact_ledger.json"}:
        raise Phase35ProtocolError("RM3-AV root artifact ledger fields changed")
    for name, digest in ledger.items():
        if not (output_root / name).is_file() or _sha(output_root / name) != digest:
            raise Phase35ProtocolError(f"RM3-AV root artifact hash mismatch: {name}")
    status = _read(output_root / "matrix_execution_status.json")
    if status.get("all_complete") is not True:
        raise Phase35ProtocolError("RM3-AV matrix is not complete")


def _verify_run(
    output_root: Path, spec: RM3AVRunSpec, required: set[str]
) -> dict[str, Any]:
    directory = output_root / spec.run_id
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != required:
        raise Phase35ProtocolError(f"RM3-AV run artifact set changed: {spec.run_id}")
    ledger = _read(directory / "artifact_ledger.json")
    if set(ledger) != required - {"artifact_ledger.json"}:
        raise Phase35ProtocolError(f"RM3-AV run ledger fields changed: {spec.run_id}")
    for name, digest in ledger.items():
        if _sha(directory / name) != digest:
            raise Phase35ProtocolError(f"RM3-AV run hash mismatch: {spec.run_id}/{name}")
    manifest = _read(directory / "manifest.json")
    metrics = _read(directory / "metrics_validation.json")
    diagnostics = _read(directory / "diagnostics_validation.json")
    if (
        manifest.get("run_id") != spec.run_id
        or metrics.get("run_id") != spec.run_id
        or diagnostics.get("candidate_id") != spec.candidate_id
    ):
        raise Phase35ProtocolError(f"RM3-AV run identity drift: {spec.run_id}")
    if any(payload.get("test_accessed") is not False for payload in (manifest, metrics, diagnostics)):
        raise Phase35ProtocolError(f"RM3-AV test access found: {spec.run_id}")
    if manifest.get("selector_reporting_disjoint") is not True:
        raise Phase35ProtocolError(f"RM3-AV selector/reporting overlap: {spec.run_id}")
    if manifest.get("selector_reporting_utc_day_disjoint") is not True:
        raise Phase35ProtocolError(f"RM3-AV selector/reporting UTC days overlap: {spec.run_id}")
    if set(diagnostics.get("manual_audit_verdicts", {})) != set(build_manual_verdict_template()):
        raise Phase35ProtocolError(f"RM3-AV verdict fields changed: {spec.run_id}")
    if any(value is not None for value in diagnostics["manual_audit_verdicts"].values()):
        raise Phase35ProtocolError(f"Linux assigned a scientific verdict: {spec.run_id}")
    return {"manifest": manifest, "metrics": metrics, "diagnostics": diagnostics}


def _mode_delta(diagnostics: Mapping[str, Any], mode: str, task: str) -> float:
    records = diagnostics["mode_records"]
    return float(records[mode][f"{task}_mae_c"] - records["normal"][f"{task}_mae_c"])


def _record(spec: RM3AVRunSpec, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    diagnostics = payload["diagnostics"]
    values = metrics["metrics"]
    response = diagnostics["response_trajectory"]
    return {
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        "fold_id": spec.fold_id,
        "terminal_mae_c": float(values["terminal_mae_c"]),
        "local_mae_c": float(values["local_mae_c"]),
        "tin_mae_c": float(values["tin_mae_c"]),
        "valve_mae": float(values["valve_mae"]),
        "terminal_persistence_skill": diagnostics["terminal"]["skill_vs_persistence_pooled"],
        "local_persistence_skill": diagnostics["local"]["skill_vs_persistence_pooled"],
        "valve_persistence_skill": diagnostics["valve_trajectory"]["persistence_skill"],
        "response_abs_h60": float(response["mean_absolute_effect_by_horizon_steps"]["60"]),
        "explicit_to_total_local_change_ratio": response.get("explicit_to_total_local_change_ratio"),
        "response_off_terminal_mae_delta": _mode_delta(diagnostics, "response_off", "terminal"),
        "bypass_off_terminal_mae_delta": _mode_delta(diagnostics, "bypass_off", "terminal"),
        "wrong_side_local_mae_delta": _mode_delta(diagnostics, "wrong_side", "local"),
        "lead_local_mae_delta": _mode_delta(diagnostics, "lead", "local"),
        "response_training_path_reachable": diagnostics["training_graph"]["response_training_path_reachable"],
        "optimizer_updates_completed": int(metrics["optimizer_updates_completed"]),
    }


def _paired_contrasts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["candidate_id"], row["fold_id"]): row for row in records}
    fields = (
        "terminal_mae_c", "local_mae_c", "tin_mae_c", "valve_mae",
        "valve_persistence_skill", "response_abs_h60",
        "explicit_to_total_local_change_ratio", "response_off_terminal_mae_delta",
        "bypass_off_terminal_mae_delta", "wrong_side_local_mae_delta",
        "lead_local_mae_delta",
    )
    result = []
    for candidate, baseline in PAIR_BASELINES.items():
        fold_rows = []
        for fold in ("F0", "F1"):
            candidate_row, baseline_row = by_key[(candidate, fold)], by_key[(baseline, fold)]
            deltas = {
                field: (
                    float(candidate_row[field] - baseline_row[field])
                    if candidate_row[field] is not None and baseline_row[field] is not None
                    else None
                )
                for field in fields
            }
            fold_rows.append({"fold_id": fold, "candidate_minus_baseline": deltas})
        mean_delta = {
            field: (
                float(np.mean([
                    row["candidate_minus_baseline"][field] for row in fold_rows
                    if row["candidate_minus_baseline"][field] is not None
                ]))
                if any(row["candidate_minus_baseline"][field] is not None for row in fold_rows)
                else None
            )
            for field in fields
        }
        result.append({
            "candidate_id": candidate,
            "baseline_candidate_id": baseline,
            "fold_count": 2,
            "fold_deltas": fold_rows,
            "mean_candidate_minus_baseline": mean_delta,
            "lower_is_better_only_for_mae_fields": True,
            "scientific_verdict": None,
        })
    return result


def _initialization_audit(payloads: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    mismatch_count = 0
    for candidate, modules in EXPECTED_SHARED_INITIALIZATION.items():
        baseline = PAIR_BASELINES[candidate]
        for fold in ("F0", "F1"):
            candidate_hashes = payloads[(candidate, fold)]["manifest"]["initialization_hashes"]
            baseline_hashes = payloads[(baseline, fold)]["manifest"]["initialization_hashes"]
            equality = {module: candidate_hashes[module] == baseline_hashes[module] for module in modules}
            mismatch_count += sum(not value for value in equality.values())
            rows.append({
                "candidate_id": candidate,
                "baseline_candidate_id": baseline,
                "fold_id": fold,
                "expected_shared_modules": list(modules),
                "hash_equal": equality,
                "all_expected_shared_equal": all(equality.values()),
            })
    for candidate, baseline in (("C25", "C26"), ("C28", "C29")):
        modules = ("encoder", "valve_policy", "tin", "free_residual", "downstream")
        for fold in ("F0", "F1"):
            candidate_hashes = payloads[(candidate, fold)]["manifest"]["initialization_hashes"]
            baseline_hashes = payloads[(baseline, fold)]["manifest"]["initialization_hashes"]
            equality = {
                module: candidate_hashes[module] == baseline_hashes[module]
                for module in modules
            }
            mismatch_count += sum(not value for value in equality.values())
            rows.append({
                "candidate_id": candidate,
                "baseline_candidate_id": baseline,
                "fold_id": fold,
                "comparison_role": "cross_architecture_shared_module_rng_audit",
                "expected_shared_modules": list(modules),
                "hash_equal": equality,
                "all_expected_shared_equal": all(equality.values()),
            })
    return {
        "comparison_count": len(rows),
        "mismatch_count": mismatch_count,
        "all_expected_shared_modules_equal": mismatch_count == 0,
        "rows": rows,
    }


def build_av2_audit(output_root: Path, matrix: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Verify all 64 returned runs and assemble evidence without assigning verdicts."""

    validate_rm3av_matrix(matrix, repo_root=repo_root)
    run_required = set(matrix["execution_contract"]["required_run_artifacts"])
    root_required = set(matrix["execution_contract"]["required_root_artifacts"])
    _verify_root(output_root, root_required)
    specs = rm3av_run_specs(matrix, repo_root=repo_root)
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    records = []
    for spec in specs:
        payload = _verify_run(output_root, spec, run_required)
        payloads[(spec.candidate_id, spec.fold_id)] = payload
        records.append(_record(spec, payload))
    if len(records) != 64:
        raise Phase35ProtocolError("RM3-AV2 requires exactly 64 complete units")
    evidence_index = {
        question: {
            "sources": sources,
            "verdict": None,
            "allowed_verdicts": ["SUPPORTED", "REFUTED", "MIXED", "NOT_TESTABLE"],
        }
        for question, sources in QUESTION_EVIDENCE.items()
    }
    if set(evidence_index) != set(build_manual_verdict_template()):
        raise Phase35ProtocolError("RM3-AV2 evidence index does not cover Q01-Q33")
    return {
        "protocol_version": "phase3.5-ms3r-rm3av2-v1",
        "scope": "cache_free_supervisor_evidence_assembly_not_automatic_scientific_decision",
        "artifact_integrity_pass": True,
        "training_unit_count": 64,
        "candidate_count": 32,
        "folds": ["F0", "F1"],
        "paired_contrasts": _paired_contrasts(records),
        "initialization_fairness": _initialization_audit(payloads),
        "question_evidence_index": evidence_index,
        "manual_audit_verdicts": build_manual_verdict_template(),
        "rm3b_input_manifest": None,
        "rm3b_authorized": False,
        "model_champion": None,
        "composite_ranking": None,
        "test_accessed": False,
        "automatic_scientific_pass": None,
        "records": records,
    }
