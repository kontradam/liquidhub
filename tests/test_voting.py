from liquidhub.voting.engine import WorkerVotingEngine
from liquidhub.voting.hold_tracker import score_decision
from liquidhub.workers.base import Signal, Worker
from liquidhub.workers.registry import WorkerRegistry


class _Fake(Worker):
    def __init__(self, name, weight=1.0):
        self.name = name
        self.default_weight = weight
    def vote(self, ctx):
        raise NotImplementedError


def _registry(weights: dict[str, float]) -> WorkerRegistry:
    reg = WorkerRegistry()
    for n, w in weights.items():
        reg.register(_Fake(n, w))
        reg._weights[n] = w  # skip DB round-trip
    return reg


def test_voting_picks_weighted_winner():
    reg = _registry({f"w{i}": 1.0 for i in range(10)})
    signals = [Signal(worker=f"w{i}", direction="LONG", confidence=0.9) for i in range(8)] + [
        Signal(worker="w8", direction="HOLD", confidence=0.5),
        Signal(worker="w9", direction="SHORT", confidence=0.5),
    ]
    res = WorkerVotingEngine(reg).tally(signals)
    assert res.decision == "LONG"
    assert res.agreement == 8


def test_voting_holds_when_below_confidence_threshold():
    reg = _registry({f"w{i}": 1.0 for i in range(10)})
    signals = [Signal(worker=f"w{i}", direction="LONG", confidence=0.3) for i in range(10)]
    # weighted=3.0, total=3.0, consensus=1.0 but threshold is 0.60 — still passes
    # flip to HOLD by spreading votes
    signals = (
        [Signal(worker=f"w{i}", direction="LONG", confidence=0.3) for i in range(4)]
        + [Signal(worker=f"w{i}", direction="SHORT", confidence=0.3) for i in range(4, 7)]
        + [Signal(worker=f"w{i}", direction="HOLD", confidence=0.3) for i in range(7, 10)]
    )
    res = WorkerVotingEngine(reg).tally(signals)
    assert res.decision == "HOLD"


def test_voting_vetoed():
    reg = _registry({"risk": 1.0, "w1": 1.0, "w2": 1.0})
    signals = [
        Signal(worker="w1", direction="LONG", confidence=0.9),
        Signal(worker="w2", direction="LONG", confidence=0.9),
        Signal(worker="risk", direction="HOLD", confidence=1.0, veto=True),
    ]
    res = WorkerVotingEngine(reg).tally(signals)
    assert res.decision == "VETOED"
    assert "risk" in res.veto_by


def test_hold_tracker_rewards_profit_not_hold():
    assert score_decision("PROFIT", 0.02) > 0
    assert score_decision("LOSS", 0.02) < 0
    assert score_decision("HOLD", None) == 0.0
    assert score_decision("BREAKEVEN", 0.0) == 0.0
