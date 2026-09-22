import typer

from liquidhub import __version__
from liquidhub.logging import get_logger

app = typer.Typer(help="LiquidhubIndicatorSystem CLI", no_args_is_help=True)
log = get_logger("cli")


@app.command()
def version() -> None:
    """Print version."""
    typer.echo(f"liquidhub {__version__}")


@app.command()
def collect(
    timeframe: str = typer.Option("5m,15m,1h,4h", help="Comma-separated timeframes"),
    once: bool = typer.Option(False, help="Single fetch instead of continuous loop"),
) -> None:
    """Collect OHLCV candles from the configured exchange into Postgres."""
    from liquidhub.data.ohlcv_collector import run_collector

    run_collector(timeframes=[t.strip() for t in timeframe.split(",")], once=once)


@app.command()
def backfill(
    timeframe: str = typer.Option("5m,15m,1h,4h", help="Comma-separated timeframes"),
    days: int = typer.Option(30, help="Lookback window in days"),
) -> None:
    """Walk active pairs, find OHLCV gaps, and fetch missing candles."""
    from liquidhub.data.ohlcv_backfill import run_backfill

    run_backfill(timeframes=[t.strip() for t in timeframe.split(",")], lookback_days=days)


@app.command()
def run() -> None:
    """Run the paper trading orchestrator (main 15-min loop)."""
    from liquidhub.paper.orchestrator import main_loop

    main_loop()


@app.command()
def dashboard(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
) -> None:
    """Launch the FastAPI dashboard."""
    import uvicorn

    from liquidhub.config import settings

    uvicorn.run(
        "liquidhub.dashboard.api:app",
        host=host or settings.dashboard_host,
        port=port or settings.dashboard_port,
        reload=False,
    )


@app.command()
def backtest(
    symbol: str = typer.Argument(...),
    timeframe: str = typer.Option("15m"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Run a backtest for a symbol/timeframe window."""
    log.info("backtest stub: %s %s %s..%s", symbol, timeframe, start, end)


@app.command("db-init")
def db_init() -> None:
    """Create all tables (dev convenience; use alembic in prod)."""
    from liquidhub.db.engine import create_all

    create_all()
    typer.echo("ok")


@app.command()
def score() -> None:
    """Score mature shadow trades and adjust worker weights."""
    from liquidhub.feedback.scorer import score_mature_trades
    from liquidhub.feedback.loop import reward_punish
    from liquidhub.workers.registry import WorkerRegistry

    n = score_mature_trades()
    typer.echo(f"scored {n} shadow trades")
    if n > 0:
        reg = WorkerRegistry()
        reg.load_weights_from_db()
        adj = reward_punish(reg)
        typer.echo(f"adjusted {adj} worker weights")


if __name__ == "__main__":
    app()
