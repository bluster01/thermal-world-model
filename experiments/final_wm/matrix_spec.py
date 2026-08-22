"""Frozen discrimination-matrix specifications (execution mirror).

This module is the executable twin of
`docs/plans/2026-08-18-final-wm-discrimination-matrix.md`.  Linux runs it
verbatim; thresholds and arm definitions are frozen here so the executed
matrix cannot drift from the document.  Any change requires a new local
freeze and a new commit before execution.
"""

from __future__ import annotations

from dataclasses import replace

from src.final_wm.training import TrainSpec

MATRIX_VERSION = "0.2"  # v0.2: uniform T1 training budget amendment (see matrix doc §5)
SEEDS = (0, 1, 2)
HISTORY_STEPS = 96
HORIZON = 18

# Frozen verdict thresholds (matrix document §2/§3).
THRESH_O1_NLL = 0.05        # learned/hybrid vs steady, H18 NLL relative improvement
THRESH_T1_NLL = 0.02        # nested structure gains, H18 NLL
THRESH_B1_CRPS = 0.03       # boundary model vs persistence, H18 CRPS
THRESH_J1_NLL = 0.03        # joint vs staged, H18 NLL
THRESH_R1_LEAK = 0.05       # aware-probe relative gain over blind probe
MIN_SEED_PASSES = 2         # >=2/3 seeds must pass


def _base(unit: str, arm: str, seed: int, **kw) -> TrainSpec:
    return TrainSpec(unit=unit, arm=arm, seed=seed, history_steps=HISTORY_STEPS, horizon=HORIZON, **kw)


def o1_specs(seeds: tuple[int, ...] = SEEDS) -> list[TrainSpec]:
    return [
        _base("o1", arm, seed, boundary_mode="oracle", initial_state_mode=mode)
        for seed in seeds
        for arm, mode in (("steady", "steady"), ("learned", "learned"), ("hybrid", "hybrid"))
    ]


def t1_specs(seeds: tuple[int, ...] = SEEDS) -> list[TrainSpec]:
    """Nested structure arms.  v0.2 amendment: uniform budget epochs=60 /
    patience=10 for ALL T1 arms (default 30/6 undertrained the latent arm on
    side A seed 2 — it hit the epoch cap while still descending; a uniform
    raise keeps the comparison symmetric, early stopping still bounds cost).
    """
    arms = (
        ("physics_only", "none", 0),
        ("closure_cons", "conservative", 0),
        ("closure_steam", "steam_only", 0),
        ("latent4", "conservative", 4),
        # Amendment v0.4 (audit F3): first-class rewetting ablation arm.
        # Evidence arm only -- it is NOT in the frozen T1 verdict `nested`
        # pairs; its adjudication is the v0.4 pre-registered decision rule.
        ("closure_cons_norew", "conservative_norew", 0),
    )
    return [
        _base("t1", arm, seed, boundary_mode="oracle", initial_state_mode="hybrid",
              closure_mode=closure, latent_dim=latent, epochs=60, patience=10)
        for seed in seeds
        for arm, closure, latent in arms
    ]


def b1_specs(seeds: tuple[int, ...] = SEEDS) -> list[TrainSpec]:
    return [
        _base("b1", "gru", seed, boundary_mode="forecast", boundary_loss_only=True)
        for seed in seeds
    ]


def j1_specs(seeds: tuple[int, ...] = SEEDS) -> list[TrainSpec]:
    """Joint vs staged.  The structure is frozen at the T1 default winner
    (conservative closure, hybrid init); if T1 rejects the closure the J1
    submission is re-frozen before execution."""
    specs: list[TrainSpec] = []
    for seed in seeds:
        specs.append(
            _base("j1", "joint", seed, boundary_mode="forecast", train_boundary=True,
                  initial_state_mode="hybrid", closure_mode="conservative")
        )
        specs.append(
            _base("j1", "staged_main", seed, boundary_mode="oracle",
                  initial_state_mode="hybrid", closure_mode="conservative")
        )
    return specs


def j1_staged_boundary_spec(seed: int, main_checkpoint: str) -> TrainSpec:
    return _base("j1", f"staged_boundary_from_{seed}", seed, boundary_mode="forecast",
                 boundary_loss_only=True, init_checkpoint=main_checkpoint)


def quicken(spec: TrainSpec) -> TrainSpec:
    """Tiny sizes for local smoke / Linux dry-run.  Never used for verdicts."""
    return replace(spec, epochs=2, batches_per_epoch=4, batch_size=8, eval_windows=16, eval_batch=8)
