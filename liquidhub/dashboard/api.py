from datetime import UTC, datetime

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Integer, desc, func, select

from liquidhub import __version__
from liquidhub.config import settings
from liquidhub.db.engine import session_scope
from liquidhub.db.models import OHLCV, MarketPair, ShadowTrade, SystemEvent, WorkerRegistry, WorkerSignal
from pathlib import Path

app = FastAPI(title="LiquidhubIndicatorSystem Dashboard", version=__version__)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/status")
def status() -> dict:
    with session_scope() as s:
        worker_count = s.scalar(
            select(func.count()).select_from(WorkerRegistry).where(WorkerRegistry.enabled.is_(True))
        )
        symbol_count = s.scalar(
            select(func.count()).select_from(MarketPair).where(
                MarketPair.exchange == settings.exchange,
                MarketPair.active.is_(True),
            )
        )
    return {
        "exchange": settings.exchange,
        "timeframes": settings.timeframes,
        "worker_count": worker_count or 0,
        "symbol_count": symbol_count or 0,
        "thresholds": {
            "min_confidence": settings.vote_min_confidence,
            "min_consensus": settings.vote_min_consensus,
            "agreement_min": settings.worker_agreement_min,
        },
        "tick_interval": settings.tick_interval_seconds,
    }


@app.get("/api/workers")
def list_workers() -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(WorkerRegistry).order_by(WorkerRegistry.kind, WorkerRegistry.name)).scalars().all()
        return [
            {
                "name": r.name,
                "kind": r.kind,
                "weight": r.weight,
                "enabled": r.enabled,
                "streak": (r.meta or {}).get("streak", 0),
                "best_streak": (r.meta or {}).get("best_streak", 0),
                "worst_streak": (r.meta or {}).get("worst_streak", 0),
            }
            for r in rows
        ]


@app.get("/api/ranking/accumulation-breakout")
def ranking_accumulation_breakout() -> list[dict]:
    with session_scope() as s:
        latest_ts = s.scalar(
            select(func.max(WorkerSignal.ts))
        )
        if latest_ts is None:
            return []

        signals = s.execute(
            select(WorkerSignal).where(
                WorkerSignal.ts == latest_ts,
                WorkerSignal.direction.in_(["LONG", "SHORT"]),
            ).order_by(WorkerSignal.symbol, WorkerSignal.worker_id)
        ).scalars().all()

        by_symbol: dict[str, dict] = {}
        for sig in signals:
            if sig.symbol not in by_symbol:
                by_symbol[sig.symbol] = {
                    "symbol": sig.symbol,
                    "ts": sig.ts.isoformat() if sig.ts else None,
                    "accumulation_signals": [],
                    "breakout_signals": [],
                    "acc_avg": 0.0,
                    "brk_avg": 0.0,
                    "composite": 0.0,
                }
            entry = by_symbol[sig.symbol]
            reason = sig.reason or {}
            worker_name = _resolve_worker_name(s, sig.worker_id)
            sig_data = {
                "worker": worker_name,
                "direction": sig.direction,
                "confidence": round(sig.confidence, 3),
                "reason": reason,
            }
            if worker_name and worker_name.startswith("accumulation_"):
                entry["accumulation_signals"].append(sig_data)
            elif worker_name and worker_name.startswith("breakout_"):
                entry["breakout_signals"].append(sig_data)

        results = []
        for entry in by_symbol.values():
            acc_probs = [s["confidence"] for s in entry["accumulation_signals"] if s["direction"] == "LONG"]
            brk_probs = [s["confidence"] for s in entry["breakout_signals"] if s["direction"] == "LONG"]
            entry["acc_avg"] = round(sum(acc_probs) / len(acc_probs), 3) if acc_probs else 0.0
            entry["brk_avg"] = round(sum(brk_probs) / len(brk_probs), 3) if brk_probs else 0.0
            entry["composite"] = round(entry["acc_avg"] * 0.5 + entry["brk_avg"] * 0.5, 3)
            results.append(entry)

        results.sort(key=lambda x: x["composite"], reverse=True)
        return results


def _resolve_worker_name(s, worker_id: int) -> str:
    from sqlalchemy.orm import Session
    row = s.get(WorkerRegistry, worker_id)
    return row.name if row else f"worker_{worker_id}"


@app.get("/api/ranking/voting")
def ranking_voting() -> list[dict]:
    with session_scope() as s:
        latest_ts = s.scalar(select(func.max(WorkerSignal.ts)))
        if latest_ts is None:
            return []

        signals = s.execute(
            select(WorkerSignal).where(
                WorkerSignal.ts == latest_ts,
            ).order_by(WorkerSignal.symbol)
        ).scalars().all()

        by_symbol: dict[str, dict] = {}
        for sig in signals:
            if sig.symbol not in by_symbol:
                by_symbol[sig.symbol] = {
                    "symbol": sig.symbol,
                    "ts": sig.ts.isoformat() if sig.ts else None,
                    "directional": [],
                    "forecast": [],
                    "winner": "LONG",
                    "mark": 0.0,
                    "consensus": 0.0,
                    "strength": 0.0,
                    "agreement": 0,
                }

            worker_name = _resolve_worker_name(s, sig.worker_id)
            reason = sig.reason or {}
            sig_data = {
                "worker": worker_name,
                "direction": sig.direction,
                "confidence": round(sig.confidence, 3),
                "reason": reason,
            }

            entry = by_symbol[sig.symbol]
            if worker_name and (worker_name.startswith("accumulation_") or worker_name.startswith("breakout_") or worker_name.startswith("btc_cycle")):
                entry["forecast"].append(sig_data)
            else:
                entry["directional"].append(sig_data)

        directional_workers = _get_directional_worker_names()

        for entry in by_symbol.values():
            long_w = 0.0
            short_w = 0.0
            long_c = 0
            short_c = 0
            for sig_data in entry["directional"]:
                w = _get_worker_weight(s, sig_data["worker"])
                if sig_data["direction"] == "LONG":
                    long_w += w * sig_data["confidence"]
                    long_c += 1
                else:
                    short_w += w * sig_data["confidence"]
                    short_c += 1

            total = long_w + short_w or 1.0
            winner = "LONG" if long_w >= short_w else "SHORT"
            consensus = max(long_w, short_w) / total
            strength = abs(long_w - short_w) / total
            mark = consensus * strength

            entry["winner"] = winner
            entry["mark"] = round(mark, 4)
            entry["consensus"] = round(consensus, 3)
            entry["strength"] = round(strength, 3)
            entry["agreement"] = long_c if winner == "LONG" else short_c

        results = list(by_symbol.values())
        results.sort(key=lambda x: x["mark"], reverse=True)
        return results


def _get_directional_worker_names() -> set[str]:
    from liquidhub.workers.accumulation_breakout import AccumulationWorker, BreakoutWorker
    from liquidhub.workers.bb import BollingerBandsWorker
    from liquidhub.workers.btc_cycle import BTCCycleWorker
    from liquidhub.workers.ma import MAWorker
    from liquidhub.workers.macd import MACDWorker
    from liquidhub.workers.ml_xgboost import XGBoostWorker
    from liquidhub.workers.momentum import MomentumWorker
    from liquidhub.workers.rsi import RSIWorker
    from liquidhub.workers.volume import VolumeWorker
    from liquidhub.ml.models_registry import available_timeframes

    names = set()
    for tf in available_timeframes("xgboost"):
        if tf in settings.timeframes:
            names.add(f"xgboost_{tf}_v1")
    for W in (RSIWorker, MACDWorker, MAWorker, BollingerBandsWorker, VolumeWorker, MomentumWorker):
        names.add(W.name)
    return names


def _get_worker_weight(s, worker_name: str) -> float:
    row = s.execute(select(WorkerRegistry).where(WorkerRegistry.name == worker_name)).scalar_one_or_none()
    return row.weight if row else 1.0


@app.get("/api/leaderboard")
def leaderboard() -> list[dict]:
    with session_scope() as s:
        latest_ts = s.scalar(select(func.max(WorkerSignal.ts)))
        if latest_ts is None:
            return []

        rows = s.execute(
            select(WorkerSignal).where(
                WorkerSignal.ts == latest_ts,
            ).order_by(WorkerSignal.worker_id, WorkerSignal.symbol)
        ).scalars().all()

        by_worker: dict[str, dict] = {}
        for sig in rows:
            worker_name = _resolve_worker_name(s, sig.worker_id)
            if worker_name not in by_worker:
                row = s.execute(select(WorkerRegistry).where(WorkerRegistry.name == worker_name)).scalar_one_or_none()
                meta = (row.meta or {}) if row else {}
                by_worker[worker_name] = {
                    "name": worker_name,
                    "kind": row.kind if row else "unknown",
                    "weight": round(row.weight, 2) if row else 1.0,
                    "enabled": row.enabled if row else True,
                    "streak": int(meta.get("streak", 0)),
                    "best_streak": int(meta.get("best_streak", 0)),
                    "worst_streak": int(meta.get("worst_streak", 0)),
                    "symbols_count": 0,
                    "long_count": 0,
                    "short_count": 0,
                    "avg_confidence": 0.0,
                    "weighted_score": 0.0,
                    "long_pct": 0.0,
                    "symbols": [],
                }
            entry = by_worker[worker_name]
            entry["symbols_count"] += 1
            if sig.direction == "LONG":
                entry["long_count"] += 1
            else:
                entry["short_count"] += 1
            entry["weighted_score"] += sig.confidence * (entry["weight"] or 1.0)
            entry["symbols"].append(sig.symbol)

        for entry in by_worker.values():
            total = entry["long_count"] + entry["short_count"] or 1
            entry["long_pct"] = round(entry["long_count"] / total * 100, 1)
            entry["avg_confidence"] = round(entry["weighted_score"] / total, 3)
            entry["weighted_score"] = round(entry["weighted_score"], 3)
            del entry["symbols"]

        results = list(by_worker.values())
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results


@app.get("/api/events/recent")
def recent_events(limit: int = Query(50, le=200)) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(
            select(SystemEvent).order_by(desc(SystemEvent.ts)).limit(limit)
        ).scalars().all()
        return [
            {
                "ts": r.ts.isoformat(),
                "level": r.level,
                "source": r.source,
                "message": r.message,
                "payload": r.payload,
            }
            for r in rows
        ]


@app.get("/api/signals/latest")
def signals_latest() -> list[dict]:
    """Clean per-symbol signal for downstream consumers (Aegis).

    Class A — forecast: averaged P(accumulation) and P(breakout) across
    timeframes (accumulation_*, breakout_* workers).
    Class B — directional: LONG/SHORT consensus from the top-N champion
    directional workers (ranked by weight, tie-broken by streak). HOLD if
    fewer than `signal_min_champion_agreement` champions agree.
    """
    top_n = settings.signal_top_n_champions
    min_agree = settings.signal_min_champion_agreement
    use_dir_wr = settings.signal_use_directional_winrate
    min_samples = settings.signal_min_winrate_samples
    forecast_prefixes = ("accumulation_", "breakout_", "btc_cycle")

    with session_scope() as s:
        latest_ts = s.scalar(select(func.max(WorkerSignal.ts)))
        if latest_ts is None:
            return []

        all_workers = {
            r.id: {
                "name": r.name,
                "kind": r.kind,
                "weight": float(r.weight or 0.0),
                "enabled": bool(r.enabled),
                "streak": int((r.meta or {}).get("streak", 0)),
                "best_streak": int((r.meta or {}).get("best_streak", 0)),
            }
            for r in s.execute(select(WorkerRegistry)).scalars()
        }

        dir_winrate: dict[tuple[str, str], float] = {}
        if use_dir_wr:
            for r in s.execute(
                select(
                    ShadowTrade.worker_id,
                    ShadowTrade.direction,
                    func.count().label("total"),
                    func.sum(func.cast(ShadowTrade.pnl_pct > 0, Integer)).label("wins"),
                )
                .where(ShadowTrade.status == "CLOSED", ShadowTrade.pnl_pct.isnot(None))
                .group_by(ShadowTrade.worker_id, ShadowTrade.direction)
            ).all():
                total = int(r.total or 0)
                wins = int(r.wins or 0)
                w = all_workers.get(r.worker_id)
                if not w:
                    continue
                wr = (wins / total) if total >= min_samples else 0.5
                dir_winrate[(w["name"], r.direction)] = wr

        signals = s.execute(
            select(WorkerSignal).where(WorkerSignal.ts == latest_ts)
        ).scalars().all()

        by_symbol: dict[str, dict] = {}
        for sig in signals:
            w = all_workers.get(sig.worker_id)
            if w is None or not w["enabled"]:
                continue
            wname = w["name"]
            is_forecast = any(wname.startswith(p) for p in forecast_prefixes)

            entry = by_symbol.setdefault(sig.symbol, {
                "directional": [],
                "accumulation": [],
                "breakout": [],
            })
            row = {
                "worker": wname,
                "direction": sig.direction,
                "confidence": float(sig.confidence),
                "weight": w["weight"],
                "streak": w["streak"],
                "best_streak": w["best_streak"],
                "reason": sig.reason or {},
            }
            if not is_forecast:
                entry["directional"].append(row)
            elif wname.startswith("accumulation_"):
                entry["accumulation"].append(row)
            elif wname.startswith("breakout_"):
                entry["breakout"].append(row)

        out: list[dict] = []
        for symbol, e in by_symbol.items():
            champions = sorted(
                e["directional"],
                key=lambda r: (r["weight"], r["streak"]),
                reverse=True,
            )[:top_n]

            for c in champions:
                wr = dir_winrate.get((c["worker"], c["direction"]), 0.5) if use_dir_wr else 1.0
                c["dir_winrate"] = wr
                c["effective_weight"] = c["weight"] * wr

            long_w = sum(c["effective_weight"] * c["confidence"] for c in champions if c["direction"] == "LONG")
            short_w = sum(c["effective_weight"] * c["confidence"] for c in champions if c["direction"] == "SHORT")
            long_n = sum(1 for c in champions if c["direction"] == "LONG")
            short_n = sum(1 for c in champions if c["direction"] == "SHORT")
            total = long_w + short_w

            if total <= 0 or not champions:
                direction, confidence, agreement = "HOLD", 0.0, 0
            else:
                if long_w >= short_w:
                    direction, agreement = "LONG", long_n
                    confidence = long_w / total
                else:
                    direction, agreement = "SHORT", short_n
                    confidence = short_w / total
                if agreement < min_agree:
                    direction = "HOLD"

            acc_probs = [
                float(c["reason"].get("accumulation_prob", 0.0))
                for c in e["accumulation"]
                if "accumulation_prob" in c["reason"]
            ]
            brk_probs = [
                float(c["reason"].get("breakout_prob", 0.0))
                for c in e["breakout"]
                if "breakout_prob" in c["reason"]
            ]
            p_acc = sum(acc_probs) / len(acc_probs) if acc_probs else 0.0
            p_brk = sum(brk_probs) / len(brk_probs) if brk_probs else 0.0

            forecast_by_tf: dict[str, dict] = {}
            for c in e["accumulation"]:
                tf = c["worker"].split("_", 1)[1]
                forecast_by_tf.setdefault(tf, {})["accumulation"] = round(
                    float(c["reason"].get("accumulation_prob", 0.0)), 3
                )
            for c in e["breakout"]:
                tf = c["worker"].split("_", 1)[1]
                forecast_by_tf.setdefault(tf, {})["breakout"] = round(
                    float(c["reason"].get("breakout_prob", 0.0)), 3
                )

            pair = s.execute(
                select(MarketPair).where(
                    MarketPair.exchange == settings.exchange,
                    MarketPair.symbol == symbol,
                )
            ).scalar_one_or_none()
            price = 0.0
            if pair is not None:
                last = s.execute(
                    select(OHLCV.close)
                    .where(OHLCV.pair_id == pair.id, OHLCV.timeframe == settings.shadow_score_timeframe)
                    .order_by(OHLCV.timestamp.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if last is not None:
                    price = float(last)

            out.append({
                "symbol": symbol,
                "ts": latest_ts.isoformat(),
                "price": price,
                "direction": direction,
                "confidence": round(confidence, 3),
                "agreement": agreement,
                "champions_used": len(champions),
                "champions": [
                    {
                        "worker": c["worker"],
                        "vote": c["direction"],
                        "confidence": round(c["confidence"], 3),
                        "weight": round(c["weight"], 2),
                        "streak": c["streak"],
                        "dir_winrate": round(c.get("dir_winrate", 0.5), 3),
                        "effective_weight": round(c.get("effective_weight", c["weight"]), 3),
                    }
                    for c in champions
                ],
                "forecast": {
                    "p_accumulation": round(p_acc, 3),
                    "p_breakout": round(p_brk, 3),
                    "by_timeframe": forecast_by_tf,
                },
            })

        out.sort(key=lambda x: (x["direction"] != "HOLD", x["confidence"]), reverse=True)
        return out


@app.get("/api/feedback/shadow-trades")
def shadow_trades(status: str | None = Query(None), limit: int = Query(200, le=500)) -> list[dict]:
    with session_scope() as s:
        stmt = select(ShadowTrade).order_by(desc(ShadowTrade.ts)).limit(limit)
        if status:
            stmt = stmt.where(ShadowTrade.status == status)
        rows = s.execute(stmt).scalars().all()
        results = []
        for r in rows:
            worker_name = _resolve_worker_name(s, r.worker_id)
            results.append({
                "id": r.id,
                "ts": r.ts.isoformat() if r.ts else None,
                "symbol": r.symbol,
                "worker": worker_name,
                "direction": r.direction,
                "entry_price": round(r.entry_price, 6) if r.entry_price else None,
                "exit_price": round(r.exit_price, 6) if r.exit_price else None,
                "pnl_pct": round(r.pnl_pct, 6) if r.pnl_pct is not None else None,
                "confidence": round(r.confidence, 3),
                "timeframe": r.timeframe,
                "status": r.status,
            })
        return results


@app.get("/api/feedback/worker-accuracy")
def worker_accuracy() -> dict:
    """Per-worker accuracy split by direction.

    Returns:
        {
          "long_specialists": [...],   sorted by win_rate desc, only direction=LONG
          "short_specialists": [...],  sorted by win_rate desc, only direction=SHORT
          "overall": [...],            combined (legacy compatibility)
        }
    A worker can appear in both lists with different win_rates — useful when
    a worker is great at calling LONG but bad at SHORT (or vice versa).
    """
    with session_scope() as s:
        rows = s.execute(
            select(
                ShadowTrade.worker_id,
                ShadowTrade.direction,
                func.count().label("total"),
                func.sum(func.cast(ShadowTrade.pnl_pct > 0, Integer)).label("wins"),
                func.avg(ShadowTrade.pnl_pct).label("avg_pnl"),
            )
            .where(ShadowTrade.status == "CLOSED", ShadowTrade.pnl_pct.isnot(None))
            .group_by(ShadowTrade.worker_id, ShadowTrade.direction)
        ).all()

        worker_meta: dict[int, dict] = {}
        for r in s.execute(select(WorkerRegistry)).scalars():
            worker_meta[r.id] = {
                "name": r.name,
                "weight": round(float(r.weight or 0), 2),
                "streak": int((r.meta or {}).get("streak", 0)),
                "best_streak": int((r.meta or {}).get("best_streak", 0)),
            }

        long_list: list[dict] = []
        short_list: list[dict] = []
        overall_acc: dict[int, dict] = {}

        for row in rows:
            wm = worker_meta.get(row.worker_id, {"name": f"worker_{row.worker_id}", "weight": 0, "streak": 0, "best_streak": 0})
            total = row.total or 0
            wins = row.wins or 0
            losses = total - wins
            entry = {
                "worker": wm["name"],
                "weight": wm["weight"],
                "streak": wm["streak"],
                "best_streak": wm["best_streak"],
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / total, 3) if total else 0.0,
                "avg_pnl": round(float(row.avg_pnl), 6) if row.avg_pnl else 0.0,
            }
            if row.direction == "LONG":
                long_list.append(entry)
            elif row.direction == "SHORT":
                short_list.append(entry)

            agg = overall_acc.setdefault(row.worker_id, {**wm, "total_trades": 0, "wins": 0, "pnl_sum": 0.0})
            agg["total_trades"] += total
            agg["wins"] += wins
            agg["pnl_sum"] += float(row.avg_pnl or 0) * total

        overall_list = []
        for wid, agg in overall_acc.items():
            t = agg["total_trades"]
            overall_list.append({
                "worker": agg["name"],
                "weight": agg["weight"],
                "streak": agg["streak"],
                "best_streak": agg["best_streak"],
                "total_trades": t,
                "wins": agg["wins"],
                "losses": t - agg["wins"],
                "win_rate": round(agg["wins"] / t, 3) if t else 0.0,
                "avg_pnl": round(agg["pnl_sum"] / t, 6) if t else 0.0,
            })

        long_list.sort(key=lambda x: (x["win_rate"], x["total_trades"]), reverse=True)
        short_list.sort(key=lambda x: (x["win_rate"], x["total_trades"]), reverse=True)
        overall_list.sort(key=lambda x: x["win_rate"], reverse=True)

        return {
            "long_specialists": long_list,
            "short_specialists": short_list,
            "overall": overall_list,
        }
