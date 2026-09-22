# Migrations

Bootstrap:

```
uv run alembic init migrations
# edit alembic.ini: sqlalchemy.url = ${DATABASE_URL}
# edit env.py: from liquidhub.db.models import Base; target_metadata = Base.metadata
uv run alembic revision --autogenerate -m "initial"
uv run alembic upgrade head
```

For dev convenience: `uv run liquidhub db-init`.
