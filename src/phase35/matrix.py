"""Deterministic experiment-matrix loading and expansion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import ExperimentConfig, Phase35ProtocolError


@dataclass(frozen=True)
class MatrixRun:
    side: str
    seed: int
    config: ExperimentConfig

    @property
    def run_id(self) -> str:
        return f"{self.side}_{self.config.config_id}_s{self.seed}"


def load_matrix(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    required = {"protocol_version", "defaults", "experiments", "sides", "seeds"}
    missing = sorted(required - set(raw))
    if missing:
        raise Phase35ProtocolError(f"matrix missing keys: {missing}")
    if raw["sides"] != ["A", "B"]:
        raise Phase35ProtocolError("Phase 3.5 core matrix must keep A/B as separate sides")
    if not raw["seeds"]:
        raise Phase35ProtocolError("matrix seeds cannot be empty")
    return raw


def experiment_configs(matrix: dict) -> list[ExperimentConfig]:
    defaults = matrix["defaults"]
    configs = []
    seen = set()
    for override in matrix["experiments"]:
        merged = {**defaults, **override}
        config = ExperimentConfig.from_mapping(merged)
        if config.config_id in seen:
            raise Phase35ProtocolError(f"duplicate config_id={config.config_id!r}")
        seen.add(config.config_id)
        configs.append(config)
    return configs


def get_experiment_config(matrix: dict, config_id: str) -> ExperimentConfig:
    matches = [c for c in experiment_configs(matrix) if c.config_id == config_id]
    if len(matches) != 1:
        raise Phase35ProtocolError(f"matrix contains no unique config_id={config_id!r}")
    return matches[0]


def expand_matrix(matrix: dict, seeds: list[int] | None = None) -> list[MatrixRun]:
    """Expand the frozen model/side matrix for development or final seed sets.

    The model configurations and A/B sides always come from the versioned matrix.
    A caller may supply an additional seed set for a pre-registered final batch
    without duplicating or editing the development matrix.
    """
    seed_values = matrix["seeds"] if seeds is None else seeds
    if not seed_values:
        raise Phase35ProtocolError("expanded seed set cannot be empty")
    return [
        MatrixRun(side=side, seed=int(seed), config=config)
        for side in matrix["sides"]
        for config in experiment_configs(matrix)
        for seed in seed_values
    ]
