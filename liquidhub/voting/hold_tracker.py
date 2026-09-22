from liquidhub.logging import get_logger

log = get_logger("hold_tracker")

# Profit-based rewards (fixes "HOLD = safe" bug from 12_IMMEDIATE_NEXT_STEPS.md Fix 1).
# A HOLD decision earns ZERO — not +0.10 — so workers aren't incentivized to sit on cash.
# PROFIT returns a positive reward scaled by realized PnL. LOSS returns a smaller penalty.


def score_decision(outcome: str, pnl_pct: float | None) -> float:
    """Return the weight delta for a single closed decision.

    outcome: "PROFIT" | "LOSS" | "HOLD" | "BREAKEVEN"
    pnl_pct: realized return as a fraction (e.g. 0.02 for +2%). Ignored for HOLD.
    """
    if outcome == "PROFIT" and pnl_pct is not None:
        return pnl_pct * 10.0
    if outcome == "LOSS" and pnl_pct is not None:
        return -abs(pnl_pct) * 5.0
    return 0.0
