from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from liquidhub.db.engine import session_scope
from liquidhub.db.models import PaperTrade
from liquidhub.logging import get_logger

log = get_logger("paper.engine")

COMMISSION = 0.001  # 0.1% per side


@dataclass
class Fill:
    symbol: str
    side: str
    price: float
    size: float


class PaperTradingEngine:
    """Records paper trades in the DB. No market impact, fixed commission."""

    def open(self, symbol: str, side: str, price: float, size: float, leverage: float = 1.0,
             reason: dict | None = None, strategy: str = "directional") -> int:
        commission = price * size * COMMISSION
        with session_scope() as s:
            trade = PaperTrade(
                symbol=symbol,
                side=side,
                entry_price=price,
                size=size,
                leverage=leverage,
                commission=commission,
                entry_time=datetime.now(UTC),
                status="OPEN",
                strategy=strategy,
                reason=reason,
            )
            s.add(trade)
            s.flush()
            return trade.id

    def close(self, trade_id: int, exit_price: float) -> float:
        with session_scope() as s:
            trade = s.get(PaperTrade, trade_id)
            if trade is None or trade.status != "OPEN":
                raise ValueError(f"trade {trade_id} not open")
            sign = 1 if trade.side == "LONG" else -1
            gross = (exit_price - float(trade.entry_price)) * float(trade.size) * sign
            exit_commission = exit_price * float(trade.size) * COMMISSION
            pnl = gross - float(trade.commission or 0) - exit_commission
            trade.exit_price = exit_price
            trade.exit_time = datetime.now(UTC)
            trade.pnl = pnl
            trade.commission = float(trade.commission or 0) + exit_commission
            trade.status = "CLOSED"
            return pnl

    def open_positions(self, symbol: str | None = None) -> list[PaperTrade]:
        with session_scope() as s:
            stmt = select(PaperTrade).where(PaperTrade.status == "OPEN")
            if symbol:
                stmt = stmt.where(PaperTrade.symbol == symbol)
            return list(s.execute(stmt).scalars())
