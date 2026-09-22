from dataclasses import dataclass
from pathlib import Path

from liquidhub.config import settings

MODEL_VERSION = "v1_20260224"

ALL_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")

CYCLE_CONFIGS = {
    "cycle4_15m": {"timeframe": "15m", "cycle_start": "2021-11-11", "avg_duration_days": 1246, "ath": 126272.0},
}


@dataclass(frozen=True)
class ModelPaths:
    model: Path
    imputer: Path
    metadata: Path


def model_paths(timeframe: str, kind: str = "xgboost") -> ModelPaths:
    base = Path(settings.models_dir).expanduser().resolve()
    if kind == "xgboost":
        stem = f"{kind}_{timeframe}_{MODEL_VERSION}"
        return ModelPaths(
            model=base / f"{stem}.pkl",
            imputer=base / f"{stem}_imputer.pkl",
            metadata=base / f"{stem}.json",
        )
    if kind in ("accumulation", "breakout"):
        subdir = base / "accumulation_breakout"
        return ModelPaths(
            model=subdir / f"{kind}_{timeframe}.pkl",
            imputer=subdir / f"{kind}_{timeframe}_imputer.pkl",
            metadata=subdir / f"{kind}_{timeframe}_meta.json",
        )
    if kind == "cycle":
        cycle_dir = base / "btc_cycle"
        name = f"xgboost_{timeframe}"
        return ModelPaths(
            model=cycle_dir / f"{name}_v1.pkl",
            imputer=cycle_dir / f"{name}_v1_imputer.pkl",
            metadata=cycle_dir / f"{name}_v1.json",
        )
    raise ValueError(f"unknown model kind: {kind}")


def available_timeframes(kind: str = "xgboost") -> list[str]:
    found: list[str] = []
    for tf in ALL_TIMEFRAMES:
        try:
            p = model_paths(tf, kind)
            if p.model.exists():
                found.append(tf)
        except ValueError:
            break
    return found


def available_cycle_models() -> list[str]:
    base = Path(settings.models_dir).expanduser().resolve() / "btc_cycle"
    found: list[str] = []
    for name in CYCLE_CONFIGS:
        if (base / f"xgboost_{name}_v1.pkl").exists():
            found.append(name)
    return found
