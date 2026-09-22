"""Pytest session setup.

The default database URL uses `postgresql+psycopg://` (psycopg v3).
Not all environments have psycopg installed, and tests never touch a
real Postgres — orchestrator e2e mocks `session_scope`, the rest don't
hit the DB at all. Pick a driver that's available without forcing a
live connection: psycopg2 only imports on first connection attempt.
"""
import os
from importlib.util import find_spec

if find_spec("psycopg") is None and find_spec("psycopg2") is not None:
    os.environ.setdefault(
        "DATABASE_URL", "postgresql+psycopg2://stub:stub@localhost:5432/stub"
    )

# SQLite makes only an INTEGER PK alias for ROWID (and therefore
# auto-increment). BigInteger maps to BIGINT, which breaks autoincrement.
# For the in-memory SQLite DBs used in tests, compile BigInteger as
# INTEGER so server-side autoincrement works.
from sqlalchemy import BigInteger  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402


@compiles(BigInteger, "sqlite")
def _sqlite_bigint_as_int(element, compiler, **kw):  # pragma: no cover — dialect glue
    return "INTEGER"
