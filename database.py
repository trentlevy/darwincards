"""
Database setup — SQLAlchemy + PostgreSQL (falls back to SQLite for local dev).
Set DATABASE_URL environment variable in Railway to your Postgres connection string.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./darwincards.db")

# SQLAlchemy requires postgresql:// not postgres:// (Railway uses the latter)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_columns(table: str, columns: dict):
    """
    Tiny idempotent migration helper: adds any of `columns` (name -> SQL type)
    that don't already exist on `table`. This app has no migration framework
    (just Base.metadata.create_all, which only creates brand-new tables, never
    alters existing ones) — for a table that already has real rows in
    production (like `jobs`), a new ORM column needs this instead, or every
    read/write of that column errors with "column does not exist". Safe to
    call on every startup: does nothing once the columns are already there.
    """
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return  # brand-new table — create_all will make it with all columns already
    existing = {c["name"] for c in inspector.get_columns(table)}
    with engine.begin() as conn:
        for name, sql_type in columns.items():
            if name in existing:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
