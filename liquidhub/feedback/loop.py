from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from liquidhub.config import settings
from liquidhub.db.engine import session_scope
from liquidhub.db.models import ShadowTrade, WorkerRegistry as WorkerRow
from liquidhub.logging import get_logger
from liquidhub.workers.registry import WorkerRegistry

log = get_logger("feedback.loop")


def reward_punish(registry: WorkerRegistry) -> int:
    """Apply chronological, streak-based exponential rewards/penalties.

    Streak persists across calls in worker_registry.meta["streak"]:
        +N = N consecutive winners, -N = N consecutive losers.
    Per-trade delta scales as base * growth^(|streak|-1), so 10 wins in a row
    produces a champion (and 10 losses kill a worker fast).
    """
    growth = settings.shadow_streak_growth
    reward = settings.shadow_reward_pct
    penalty = settings.shadow_penalty_pct
    w_min = settings.shadow_min_weight
    w_max = settings.shadow_max_weight

    adjusted = 0
    with session_scope() as s:
        trades = list(
            s.execute(
                select(ShadowTrade)
                .where(
                    ShadowTrade.status == "CLOSED",
                    ShadowTrade.scored.is_(True),
                    ShadowTrade.pnl_pct.isnot(None),
                    ShadowTrade.feedback_applied.is_(False),
                )
                .order_by(ShadowTrade.worker_id, ShadowTrade.exit_ts.asc(), ShadowTrade.id.asc())
            ).scalars()
        )

        if not trades:
            return 0

        by_worker: dict[int, list[ShadowTrade]] = {}
        for t in trades:
            by_worker.setdefault(t.worker_id, []).append(t)

        for worker_id, ws in by_worker.items():
            row = s.get(WorkerRow, worker_id)
            if row is None:
                continue

            meta = dict(row.meta or {})
            streak = int(meta.get("streak", 0))
            best_streak = int(meta.get("best_streak", 0))
            worst_streak = int(meta.get("worst_streak", 0))

            old_w = row.weight
            new_w = old_w
            wins = losses = 0

            for t in ws:
                won = t.pnl_pct > 0
                if won:
                    streak = streak + 1 if streak > 0 else 1
                    delta = reward * (growth ** (streak - 1))
                    wins += 1
                else:
                    streak = streak - 1 if streak < 0 else -1
                    delta = -penalty * (growth ** (abs(streak) - 1))
                    losses += 1

                new_w = max(w_min, min(w_max, new_w + delta))
                best_streak = max(best_streak, streak)
                worst_streak = min(worst_streak, streak)
                t.feedback_applied = True

            meta["streak"] = streak
            meta["best_streak"] = best_streak
            meta["worst_streak"] = worst_streak
            row.meta = meta
            flag_modified(row, "meta")

            registry.update_weight(
                row.name,
                new_w,
                reason=f"streak={streak} wins={wins} losses={losses} n={len(ws)}",
            )
            adjusted += 1
            log.info(
                "%s: %.2f → %.2f  (streak=%+d, wins=%d, losses=%d, n=%d)",
                row.name, old_w, new_w, streak, wins, losses, len(ws),
            )

    return adjusted
