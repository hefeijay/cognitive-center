from datetime import datetime
from typing import Optional, Any

from sqlalchemy import String, TIMESTAMP, text, Text, INT, BIGINT
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class AgentTask(Base):
    """
    大模型Agent运行的任务表
    """
    __tablename__ = "agent_task"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True, init=False)
    task_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    parent_task_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    input_params: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    logs: Mapped[Optional[str]] = mapped_column(Text)
    tool_calls: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    token_usage: Mapped[Optional[int]] = mapped_column(INT)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True, default=None)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')
    priority: Mapped[int] = mapped_column(INT, default=0)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False, init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
        init=False,
    )