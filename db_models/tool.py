import uuid
from sqlalchemy import Integer, String, JSON, Text as SQLText
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional
from .base import db, Base

class Tool(Base):
    __tablename__ = 'tools'
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    mode: Mapped[str] = mapped_column(String(50), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="工具的位置信息，例如模块名和函数名"
    )
    schema_def: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="工具的参数定义，例如参数名、参数类型、参数描述"
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)
    tool_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid.uuid4()), nullable=False, unique=True, index=True)
    ownership: Mapped[str] = mapped_column(String(50), nullable=False, default='default')