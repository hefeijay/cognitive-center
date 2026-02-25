from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import (
    Index,
    String,
    TIMESTAMP,
    Integer,
    Text as SQLText,
)
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class Session(Base):
    __tablename__ = "session"
    __table_args__ = (
        Index("idx_session_session_id", "session_id"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)  
    session_id: Mapped[str] = mapped_column(String(128))
    user_id: Mapped[str] = mapped_column(String(128))
    config: Mapped[Optional[str]] = mapped_column(
        SQLText
    )
    status: Mapped[Optional[str]] = mapped_column(String(50),default="deactive")
    create_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP,default=datetime.now(timezone.utc))
    session_name: Mapped[str] = mapped_column(String(128),default="new chat")
    summary: Mapped[Optional[str]] = mapped_column(String(2048), default=None)
    updated_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP,default=datetime.now(timezone.utc),onupdate=datetime.now(timezone.utc))