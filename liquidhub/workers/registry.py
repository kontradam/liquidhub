from liquidhub.db.engine import session_scope
from liquidhub.db.models import WorkerRegistry as WorkerRow
from liquidhub.logging import get_logger
from liquidhub.workers.base import Worker

log = get_logger("workers.registry")


class WorkerRegistry:
    """Holds the in-memory roster of Workers and their current weights.

    Weights persist in `worker_registry` table. On startup, workers registered
    via `register()` are upserted into the DB; their weight is read back from DB
    (DB is the source of truth, so the feedback loop's updates survive restarts).
    """

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._weights: dict[str, float] = {}

    def register(self, worker: Worker) -> None:
        self._workers[worker.name] = worker
        self._weights[worker.name] = worker.default_weight

    def load_weights_from_db(self) -> None:
        with session_scope() as s:
            for name, w in list(self._workers.items()):
                row = s.query(WorkerRow).filter_by(name=name).one_or_none()
                if row is None:
                    row = WorkerRow(name=name, kind=w.kind, weight=w.default_weight, enabled=True)
                    s.add(row)
                    s.flush()
                self._weights[name] = row.weight
        log.info("loaded weights for %d workers", len(self._workers))

    def update_weight(self, name: str, new_weight: float, reason: str | None = None) -> None:
        from liquidhub.db.models import WeightHistory

        with session_scope() as s:
            row = s.query(WorkerRow).filter_by(name=name).one()
            s.add(
                WeightHistory(
                    worker_id=row.id, old_weight=row.weight, new_weight=new_weight, reason=reason
                )
            )
            row.weight = new_weight
        self._weights[name] = new_weight

    def active(self) -> list[Worker]:
        return list(self._workers.values())

    def weight(self, name: str) -> float:
        return self._weights.get(name, 0.0)
