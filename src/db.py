"""SQLAlchemy ORM models and session management for experiment storage."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

from src.utils import DATABASE_URL, get_logger

logger = get_logger(__name__)


class JSONType(TypeDecorator):
    """Portable JSON column that stores as TEXT for SQLite compat."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return json.dumps(value) if value is not None else None

    def process_result_value(self, value, dialect):
        return json.loads(value) if value is not None else None


class Base(DeclarativeBase):
    pass


class ExperimentConfig(Base):
    __tablename__ = "experiment_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    segment = Column(String(20), nullable=False)
    control_description = Column(String(255), nullable=False)
    treatment_description = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    config_json = Column(JSONType, nullable=True)


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(120), nullable=False, index=True)
    control_conversions = Column(Integer, nullable=False)
    control_trials = Column(Integer, nullable=False)
    treatment_conversions = Column(Integer, nullable=False)
    treatment_trials = Column(Integer, nullable=False)
    prob_treatment_beats_control = Column(Float, nullable=False)
    expected_lift = Column(Float, nullable=False)
    recommended_sample_size = Column(Integer, nullable=False)
    posterior_data = Column(JSONType, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def _make_engine(url: str = DATABASE_URL):
    return create_engine(url, pool_pre_ping=True, future=True)


try:
    engine = _make_engine()
    # Probe the connection eagerly so we know immediately if Postgres is up
    with engine.connect() as _probe:
        pass
    SessionLocal: sessionmaker[Session] = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    _db_available = True
except Exception as exc:
    logger.warning("Database not reachable (%s) — running in no-DB mode.", exc)
    engine = None  # type: ignore[assignment]
    SessionLocal = None  # type: ignore[assignment]
    _db_available = False


def is_db_available() -> bool:
    return _db_available


def init_db() -> None:
    """Create all tables (idempotent)."""
    if engine is None:
        logger.error("No database engine — skipping init_db()")
        return
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")


def get_db() -> Generator[Session, None, None]:
    if SessionLocal is None:
        raise RuntimeError("Database not available")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
