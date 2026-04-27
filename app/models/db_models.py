from datetime import datetime

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from config.settings import settings


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Tables ────────────────────────────────────────────────────────────────────

class Call(Base):
    __tablename__ = "calls"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(String, unique=True, nullable=False)
    caller_phone   = Column(String, nullable=True)
    caller_name    = Column(String, nullable=True)
    caller_email   = Column(String, nullable=True)
    caller_need    = Column(Text,   nullable=True)
    interest_level = Column(String, nullable=True)   # high / medium / low
    call_start     = Column(DateTime, nullable=False)
    call_end       = Column(DateTime, nullable=True)
    call_duration  = Column(Integer, nullable=True)  # seconds
    exchange_count = Column(Integer, default=0)
    created_at     = Column(DateTime, default=datetime.utcnow)

    exchanges = relationship("Exchange", back_populates="call",
                             cascade="all, delete-orphan")


class Exchange(Base):
    __tablename__ = "exchanges"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    call_id         = Column(Integer, ForeignKey("calls.id"), nullable=False)
    exchange_number = Column(Integer, nullable=False)
    caller_message  = Column(Text, nullable=False)
    agent_reply     = Column(Text, nullable=False)
    timestamp       = Column(DateTime, default=datetime.utcnow)

    call = relationship("Call", back_populates="exchanges")


# ── Engine & Session ──────────────────────────────────────────────────────────

engine       = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(engine)