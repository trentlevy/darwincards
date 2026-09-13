"""
Database models: User, UsageLog
"""

import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active       = Column(Boolean, default=True)
    plan            = Column(String, default="free")          # "free" or "pro"
    stripe_customer_id      = Column(String, nullable=True)
    stripe_subscription_id  = Column(String, nullable=True)
    created_at      = Column(DateTime, default=datetime.datetime.utcnow)

    usage_logs = relationship("UsageLog", back_populates="user")


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="usage_logs")
