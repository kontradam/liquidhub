from liquidhub.logging import get_logger
from liquidhub.workers.registry import WorkerRegistry

log = get_logger("voting.feedback")


class FeedbackLoop:
    """Closed-loop updater: after each trade closes, nudge worker weights based on
    whether they predicted the winning direction.

    Stub for Phase 1 — wire up after the paper orchestrator is recording real trades.
    """

    def __init__(self, registry: WorkerRegistry, learning_rate: float = 0.02) -> None:
        self.registry = registry
        self.lr = learning_rate

    def apply_trade_outcome(
        self,
        votes_by_worker: dict[str, str],
        realized_direction: str,
        pnl_pct: float,
    ) -> None:
        for worker, vote in votes_by_worker.items():
            correct = vote == realized_direction
            delta = self.lr * pnl_pct * (1.0 if correct else -1.0)
            current = self.registry.weight(worker)
            new = max(0.1, min(5.0, current + delta))
            if abs(new - current) > 1e-6:
                self.registry.update_weight(
                    worker, new, reason=f"trade {realized_direction} {pnl_pct:+.4f}"
                )
        log.info("feedback applied to %d workers", len(votes_by_worker))
