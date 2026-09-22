from dataclasses import dataclass, field
from typing import Literal

from liquidhub.config import settings
from liquidhub.logging import get_logger
from liquidhub.workers.base import Signal
from liquidhub.workers.registry import WorkerRegistry

log = get_logger("voting")

Decision = Literal["LONG", "SHORT", "UNDECIDED", "VETOED"]


@dataclass(slots=True)
class VoteResult:
    decision: Decision
    winner: Literal["LONG", "SHORT"]
    consensus: float          # winner_weight / total_weight (0..1)
    strength: float           # (winner_weight - loser_weight) / total_weight (0..1)
    agreement: int            # count of workers voting for winner
    total_workers: int
    mark: float               # composite opportunity score = consensus * strength
    signals: list[Signal] = field(default_factory=list)
    veto_by: list[str] = field(default_factory=list)
    per_direction_weight: dict[str, float] = field(default_factory=dict)


class WorkerVotingEngine:
    """Binary LONG/SHORT weighted vote across directional workers.

    Directional workers always output LONG or SHORT (no HOLD).
    Consolidation forecasting (accumulation/breakout) is handled elsewhere.

    Decision rules:
        - weighted[dir] = sum(weight_w * confidence_w) for workers voting `dir`
        - winner = argmax(weighted)
        - decision = winner if consensus/strength/agreement thresholds met, else UNDECIDED
        - any veto → VETOED
    """

    def __init__(self, registry: WorkerRegistry) -> None:
        self.registry = registry

    _FORECAST_PREFIXES = ("accumulation_", "breakout_", "btc_cycle")

    def tally(self, signals: list[Signal]) -> VoteResult:
        vetoes = [s.worker for s in signals if s.veto]
        directional = [
            s for s in signals
            if not s.veto
            and s.direction in ("LONG", "SHORT")
            and not any(s.worker.startswith(p) for p in self._FORECAST_PREFIXES)
        ]

        weighted: dict[str, float] = {"LONG": 0.0, "SHORT": 0.0}
        counts: dict[str, int] = {"LONG": 0, "SHORT": 0}

        for s in directional:
            w = self.registry.weight(s.worker)
            weighted[s.direction] += w * s.confidence
            counts[s.direction] += 1

        total = sum(weighted.values()) or 1.0
        winner = "LONG" if weighted["LONG"] >= weighted["SHORT"] else "SHORT"
        loser = "SHORT" if winner == "LONG" else "LONG"

        consensus = weighted[winner] / total
        strength = (weighted[winner] - weighted[loser]) / total
        agreement = counts[winner]
        mark = consensus * strength

        if vetoes:
            decision: Decision = "VETOED"
        elif (
            consensus >= settings.vote_min_confidence
            and strength >= settings.vote_min_consensus
            and agreement >= settings.worker_agreement_min
        ):
            decision = winner  # type: ignore[assignment]
        else:
            decision = "UNDECIDED"

        return VoteResult(
            decision=decision,
            winner=winner,  # type: ignore[arg-type]
            consensus=consensus,
            strength=strength,
            agreement=agreement,
            total_workers=len(signals),
            mark=mark,
            signals=signals,
            veto_by=vetoes,
            per_direction_weight=weighted,
        )
