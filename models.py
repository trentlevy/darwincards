"""
Database models
"""
import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey
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
