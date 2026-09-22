"""
Database models
"""
import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey, Text, LargeBinary
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    """Kept for backwards compatibility with existing data."""
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active       = Column(Boolean, default=True)
    plan            = Column(String, default="free")
    stripe_customer_id      = Column(String, nullable=True)
    stripe_subscription_id  = Column(String, nullable=True)
    created_at      = Column(DateTime, default=datetime.datetime.utcnow)
    usage_logs = relationship("UsageLog", back_populates="user")


class UsageLog(Base):
    """Kept for backwards compatibility with existing data."""
    __tablename__ = "usage_logs"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="usage_logs")


class GlobalUsage(Base):
    """Tracks total estimated API spend per calendar month."""
    __tablename__ = "global_usage"
    id                 = Column(Integer, primary_key=True, index=True)
    month              = Column(String(7), unique=True, index=True)  # "2026-09"
    estimated_cost_usd = Column(Float, default=0.0)
    generation_count   = Column(Integer, default=0)
    updated_at         = Column(DateTime, default=datetime.datetime.utcnow)


class GenerationLog(Base):
    """One row per successful generation — used for IP rate limiting and cost audit."""
    __tablename__ = "generation_logs"
    id                 = Column(Integer, primary_key=True, index=True)
    ip_address         = Column(String, index=True)
    created_at         = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    estimated_cost_usd = Column(Float, default=0.0)
    input_tokens       = Column(Integer, default=0)
    output_tokens      = Column(Integer, default=0)
    model              = Column(String, default="")


class Job(Base):
    """Tracks generation jobs — stored in DB so they survive container restarts."""
    __tablename__ = "jobs"
    id           = Column(String, primary_key=True)
    status       = Column(String, default="running")   # running / review / done / error
    messages_json = Column(Text, default="[]")          # JSON array of progress strings
    apkg_data    = Column(LargeBinary, nullable=True)  # completed .apkg bytes
    error        = Column(String, nullable=True)
    created_at   = Column(DateTime, default=datetime.datetime.utcnow)

    # Added for the swipe-review step: cards are generated and held here
    # (status="review") before the user picks which to keep and the final
    # .apkg gets built from just those.
    cards_json   = Column(Text, nullable=True)  # JSON array of generated card dicts, pre-review
    tags_json    = Column(Text, nullable=True)  # JSON array of tags to apply to the final deck
    deck_name    = Column(String, nullable=True)  # deck name to use when finalizing


class NoteChunk(Base):
    """
    A chunk of a school's "shared drive" corpus — prior years' student notes
    on lectures, recycled class to class. Retrieved at generation time to
    ground card generation in what past students found worth writing down.
    """
    __tablename__ = "note_chunks"
    id          = Column(Integer, primary_key=True, index=True)
    school      = Column(String, index=True)    # e.g. "Perelman School of Medicine"
    source_doc  = Column(String, index=True)    # original filename/label — re-ingest replaces by this key
    heading     = Column(String, nullable=True) # nearest section/lecture heading, if the source had one
    chunk_index = Column(Integer, default=0)
    chunk_text  = Column(Text)
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
