from typing import Optional
from sqlalchemy import String, Text as SQLText, Integer
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class Workflow(Base):
    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)
    workflow_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(SQLText, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    input: Mapped[Optional[str]] = mapped_column(SQLText, nullable=True)
    output: Mapped[Optional[str]] = mapped_column(SQLText, nullable=True)