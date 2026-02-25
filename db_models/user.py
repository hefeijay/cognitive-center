from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import (
    Index,
    String,
    TIMESTAMP,
    Integer,
)
from sqlalchemy.dialects.mysql import  TEXT
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class User(Base):
    __tablename__ = "user"
    __table_args__ = (
        Index("idx_user_id", "user_id"),
    )
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)  
    user_id: Mapped[str] = mapped_column(String(128))
    user_name: Mapped[str] = mapped_column(String(128))
    pass_word: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(128))
    status: Mapped[Optional[str]] = mapped_column(String(50),default="deactive")
    create_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP,default=datetime.now(timezone.utc))
    
    
