"""Frozen specification for the FMTS main-steam comparison.

This is intentionally separate from experiments.final_wm.matrix_spec: the new
comparison neither mutates nor re-adjudicates the v0.7 discrimination matrix.
"""

from __future__ import annotations

from dataclasses import asdict

from src.final_wm.training import TrainSpec
from src.final_wm.data_v2 import BOUNDARY_EXT_ELEMENTS

EXPERIMENT_ID = "fmts_mainsteam_20260911"
PROTOCOL_VERSION = "0.2"
SEEDS = (0, 1, 2)
HISTORY_STEPS = 96
HORIZON = 18
EPOCHS = 120
PATIENCE = 20

COMMON_EVALUATION = {
    "extension_registry": BOUNDARY_EXT_ELEMENTS,
    "history_order": "obs5, actions2, base_boundary7, boundary_ext9",
    "train_bank_size": 20000,
    "validation_windows": 256,
    "train_manifest_seed": 15000,
    "validation_manifest_seed": 30000,
    "response_manifest_seed": 60000,
    "quality_rule": "base valid/finite full window, history extension mapping ranges, exact 10-second continuity",
    "normalization_note": "shared train-only extension stats; fixed base physical scales for hybrids; train-only base stats for blackbox",
    "future_base_note": "blackbox receives all 7 oracle base channels; action-driven physics ignores measured spray total",
    "history_base_boundary_channels": 7,
    "history_extension_channels": 9,
    "history_total_variables": 23,
    "extension_access": "past only; shared by blackbox and both hybrid observers",
    "future_extension_access": False,
    "extension_normalization": "train-only per-channel mean/std; saved with checkpoint",
    "greybox_extension_access": "not consumed; no registered physical mapping",
    "record": "canonical Side A v2; exact Linux input identity must be recorded",
    "split": "chronological 75/15/10",
    "locked_test_enabled": False,
    "reported_target": "final_outlet_temp",
    "target_channel_index": 4,
    "boundary_mode": "oracle",
    "prediction_curve": "cumulative MAE H1..H18, equal-day then equal-seed",
    "response_valves": ("valve1_position", "valve2_position"),
    "response_dose_fraction": 0.05,
    "response_horizon_steps": HORIZON,
    "response_windows": 64,
    "primary_response_population": "all registered windows; report support violations",
}

BLACKBOX_SPEC = {
    "arm": "blackbox_itransformer",
    "history_variables": 23,
    "future_covariates": 9,
    "target": "final_outlet_temp",
    "instance_normalization": "history target-channel mean",
    "d_model": 64,
    "layers": 2,
    "heads": 4,
    "dropout": 0.1,
    "steps_cap": 3000,
    "eval_every_steps": 500,
    "patience_evaluations": 3,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "loss": "main-steam MSE",
    "future_information": "recorded actions + recorded boundaries (oracle comparison)",
}


def _structured(arm: str, seed: int, **kwargs) -> TrainSpec:
    return TrainSpec(
        unit="fmts_mainsteam",
        arm=arm,
        seed=seed,
        epochs=EPOCHS,
        patience=PATIENCE,
        history_steps=HISTORY_STEPS,
        horizon=HORIZON,
        boundary_mode="oracle",
        **kwargs,
    )


def structured_specs(seeds: tuple[int, ...] = SEEDS) -> list[TrainSpec]:
    """All structured development arms; no result-dependent construction."""
    specs: list[TrainSpec] = []
    for seed in seeds:
        specs.extend([
            _structured(
                "greybox_steady_none",
                seed,
                initial_state_mode="steady",
                closure_mode="none",
                latent_dim=0,
                observer_encoder="gru",  # instantiated but bypassed by steady mode
            ),
            _structured(
                "fusion_gru_norew",
                seed,
                initial_state_mode="hybrid",
                closure_mode="conservative_norew",
                latent_dim=0,
                observer_encoder="gru",
            ),
            _structured(
                "fusion_token_xattn_norew",
                seed,
                initial_state_mode="hybrid",
                closure_mode="conservative_norew",
                latent_dim=0,
                observer_encoder="token_cross_attention",
                observer_patch_length=16,
                observer_patch_stride=8,
                observer_attention_heads=4,
                observer_attention_layers=1,
            ),
        ])
    return specs


TOKEN_ADVANCE_RULE = {
    "population": "frozen validation manifest only",
    "primary_metric": "equal-day cumulative main-steam MAE at H18",
    "requirements": (
        "all hard implementation and stability gates pass",
        "token MAE is lower than GRU in at least 2 of 3 paired seeds",
        "equal-seed token mean MAE is no greater than equal-seed GRU mean MAE",
    ),
    "not_selectors": (
        "response-curve sign, magnitude or visual appearance",
        "locked-test performance",
        "any unregistered auxiliary output",
    ),
    "fallback": "fusion_gru_norew",
    "timing": "freeze selected hybrid before any locked-test access",
}

HARD_GATES = (
    "observer reads exactly 96 past steps and accepts no future tensors",
    "zero-initialized observer mean equals the steady physical anchor exactly",
    "observer correction remains bounded by 0.1 times state scale",
    "physical transition, closure, action contract and boundary configs match GRU control",
    "all prediction and response arrays are finite",
    "identical action sequences give exactly identical paired rollout outputs",
    "support masks and violation counts are saved for both valve probes",
)


def serialized_spec() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seeds": list(SEEDS),
        "common_evaluation": COMMON_EVALUATION,
        "blackbox": BLACKBOX_SPEC,
        "structured": [asdict(spec) for spec in structured_specs()],
        "token_advance_rule": TOKEN_ADVANCE_RULE,
        "hard_gates": HARD_GATES,
    }
