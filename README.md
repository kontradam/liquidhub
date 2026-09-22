# LiquidhubIndicatorSystem — src

Fresh-build trading signal engine for the LiquidhubIndicatorSystem project.
Designed to run on HP t740 (Linux, Python 3.11).

## Bootstrap

```bash
# one-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd LiquidhubIndicatorSystem/src

cp .env.example .env
# edit DATABASE_URL, exchange credentials, etc.

uv sync
```

## Stand up Postgres + Redis

```bash
docker compose up -d postgres redis
uv run liquidhub db-init      # creates tables (dev); use alembic in prod
```

If the existing PG data dir on `/home/adas/Dysk_S/postgres/data` is an
incompatible version, bootstrap fresh and restore from the SQL dump:

```bash
# check version
sudo cat /home/adas/Dysk_S/postgres/data/PG_VERSION

# restore from backup dump
docker compose exec -T postgres psql -U liquidhub -d liquidhub \
  < ../backup/BACKUP_KRYPTO_20260206/liquidhub.sql
```

## Use

```bash
uv run liquidhub version
uv run liquidhub collect --once              # pull OHLCV
uv run liquidhub run                          # paper orchestrator
uv run liquidhub dashboard                    # http://localhost:8000
uv run liquidhub backtest BTC/USDT:USDT --timeframe 15m --start 2026-01-01 --end 2026-02-01
```

## Tests

```bash
uv run pytest
```

## Deploy on t740

```bash
sudo cp systemd/liquidhub-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now liquidhub-bot
journalctl -u liquidhub-bot -f
```

## Layout

```
liquidhub/
├── config.py, logging.py, cli.py
├── data/        # ccxt exchange + OHLCV collector
├── db/          # SQLAlchemy engine + models + migrations
├── features/    # V9 14-feature engine (matches trained XGBoost models)
├── ml/          # Inference wrapper around models/*.pkl
├── workers/     # Worker ABC + registry (concrete workers: Phase 2+)
├── voting/      # WorkerVotingEngine, HoldTracker (profit-based), feedback loop
├── strategies/  # Directional, hedge (Phase 3)
├── paper/       # Orchestrator (15m loop) + paper engine
└── dashboard/   # FastAPI + vanilla HTML (6 tabs, auto-refresh 10s)
```

See `../cloud.md` for the full port plan and phase roadmap.
