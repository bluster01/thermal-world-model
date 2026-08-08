"""Immutable Phase 3.5 data, action, split, and run contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence


DATE_COLUMN = "date"
LOAD_COLUMN = "机组负荷"
PRESSURE_COLUMN = "主蒸汽压力"
FEEDWATER_COLUMN = "主给水流量"
COAL_COLUMN = "未校正总煤量"
STEAM_COLUMN = "主蒸汽流量"
TIN2_COLUMN = "二级减温器入口温度"
TOUT2_COLUMN = "二级减温器出口温度"
TARGET_COLUMN = "末级过热器出口汽温"
MID_SP_COLUMN = "二级减温中间设定值"
COMMAND_COLUMN = "二级减温喷水调节门指令"
SP_COLUMN = "二级减温调节阀设定"
VALVE_COLUMN = "二级减温调节门阀位"

DEFAULT_HISTORY_FEATURES = (
    LOAD_COLUMN,
    PRESSURE_COLUMN,
    FEEDWATER_COLUMN,
    COAL_COLUMN,
    STEAM_COLUMN,
    TIN2_COLUMN,
    TOUT2_COLUMN,
    TARGET_COLUMN,
    MID_SP_COLUMN,
    COMMAND_COLUMN,
    SP_COLUMN,
    VALVE_COLUMN,
)

REQUIRED_COLUMNS = tuple(dict.fromkeys(DEFAULT_HISTORY_FEATURES + (VALVE_COLUMN,)))

ACTION_MODES = {
    "none",
    "delta_no_baseline",
    "delta_with_baseline",
    "absolute",
    "absolute_plus_delta",
}
OPENING_MAPS = {"identity", "equal_percentage_r50", "monotone"}


class Phase35ProtocolError(ValueError):
    """Raised when a configuration would mix estimands or leak a split."""


@dataclass(frozen=True)
class SplitSpec:
    train: float = 0.60
    validation: float = 0.20
    test: float = 0.20

    def validate(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(v <= 0 for v in values) or abs(sum(values) - 1.0) > 1e-9:
            raise Phase35ProtocolError(f"split ratios must be positive and sum to 1, got {values}")

    def bounds(self, n_rows: int) -> dict[str, tuple[int, int]]:
        self.validate()
        if n_rows < 3:
            raise Phase35ProtocolError("at least three rows are required for chronological splits")
        train_end = int(n_rows * self.train)
        val_end = train_end + int(n_rows * self.validation)
        return {
            "train": (0, train_end),
            "validation": (train_end, val_end),
            "test": (val_end, n_rows),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """One fully specified Phase 3.5 training run."""

    config_id: str
    action_mode: str
    opening_map: str = "identity"
    rate_branch: bool = False
    free_head: bool = True
    window: int = 96
    horizon: int = 60
    d_model: int = 32
    n_heads: int = 4
    dropout: float = 0.10
    loss: str = "huber"
    batch_size: int = 128
    steps_per_epoch: int = 200
    epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    freeze_free_epochs: int = 0
    max_train_anchors: int = 0
    max_eval_anchors: int = 4096
    history_features: tuple[str, ...] = DEFAULT_HISTORY_FEATURES

    def validate(self) -> None:
        if not self.config_id:
            raise Phase35ProtocolError("config_id must be non-empty")
        if self.action_mode not in ACTION_MODES:
            raise Phase35ProtocolError(f"unknown action_mode={self.action_mode!r}")
        if self.opening_map not in OPENING_MAPS:
            raise Phase35ProtocolError(f"unknown opening_map={self.opening_map!r}")
        if self.action_mode == "none" and (self.rate_branch or self.opening_map != "identity"):
            raise Phase35ProtocolError("free_only/none action cannot enable a valve map or rate branch")
        if self.opening_map != "identity" and self.action_mode == "delta_no_baseline":
            raise Phase35ProtocolError("a nonlinear absolute opening map requires the valve baseline")
        if self.rate_branch and self.action_mode != "absolute_plus_delta":
            raise Phase35ProtocolError("rate_branch is only valid for absolute_plus_delta")
        if self.loss not in {"huber", "mae", "nll"}:
            raise Phase35ProtocolError(f"unsupported loss={self.loss!r}")
        if self.window < 2 or self.horizon < 1 or self.d_model < 4:
            raise Phase35ProtocolError("window, horizon, and d_model are outside supported ranges")
        if self.d_model % self.n_heads:
            raise Phase35ProtocolError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise Phase35ProtocolError("dropout must be in [0, 1)")
        if min(self.batch_size, self.steps_per_epoch, self.epochs, self.patience) < 1:
            raise Phase35ProtocolError("training counts must be positive")
        missing = [c for c in (TARGET_COLUMN,) if c not in self.history_features]
        if missing:
            raise Phase35ProtocolError(f"history_features missing required target: {missing}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in fields(cls)}
        extra = sorted(set(raw) - known)
        if extra:
            raise Phase35ProtocolError(f"unknown experiment config keys: {extra}")
        values = dict(raw)
        if "history_features" in values:
            values["history_features"] = tuple(values["history_features"])
        obj = cls(**values)
        obj.validate()
        return obj

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["history_features"] = list(self.history_features)
        return out


def validate_columns(columns: Sequence[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise Phase35ProtocolError(f"raw/cache data missing required columns: {missing}")
    if len(columns) != len(set(columns)):
        raise Phase35ProtocolError("duplicate data column names are not allowed")


def validate_task_action(task: str, action_column: str) -> None:
    """Keep plant and supervisory estimands out of the same leaderboard."""
    expected = {"plant": VALVE_COLUMN, "supervisory": SP_COLUMN}
    if task not in expected:
        raise Phase35ProtocolError(f"unknown task={task!r}")
    if action_column != expected[task]:
        raise Phase35ProtocolError(
            f"task={task!r} requires action={expected[task]!r}, got {action_column!r}"
        )
