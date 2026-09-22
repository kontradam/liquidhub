import os

from liquidhub.config import Settings


def test_defaults_present():
    s = Settings()
    assert s.vote_min_confidence == 0.60
    assert s.worker_agreement_min == 7
    assert "BTC/USDT:USDT" in s.symbols


def test_csv_env_parsing(monkeypatch):
    monkeypatch.setenv("SYMBOLS", "AAA,BBB, CCC")
    monkeypatch.setenv("BLACKLIST_HOURS_UTC", "1,2,3")
    s = Settings(_env_file=None)
    assert s.symbols == ["AAA", "BBB", "CCC"]
    assert s.blacklist_hours_utc == [1, 2, 3]
    # cleanup implicit via monkeypatch
    assert "SYMBOLS" in os.environ
