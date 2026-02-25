#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消息队列数据库模型
用于存储和管理待处理的消息，支持多智能体协作系统的消息流转
"""

from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import (
    Index,
    String,
    TIMESTAMP,
    Integer,
    text,
    Enum as SQLEnum,
    Text as SQLText,
)
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class MessageQueue(Base):
    """
    消息队列模型
    存储待处理的消息，用于多智能体协作系统
    """
    __tablename__ = "message_queue"
    __table_args__ = (
        Index("idx_message_queue_status", "status"),
        Index("idx_message_queue_priority", "priority"),
        Index("idx_message_queue_created_at", "created_at"),
        Index("idx_message_queue_consumed_at", "consumed_at"),
        Index("idx_message_queue_message_type", "message_type"),
    )
    
    # 主键ID
    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True, 
        init=False,
        comment="主键ID"
    )
    
    # 消息唯一标识
    message_id: Mapped[str] = mapped_column(
        String(128), 
        unique=True, 
        nullable=False,
        comment="消息唯一标识"
    )
    
    # 消息内容
    content: Mapped[str] = mapped_column(
        SQLText,
        nullable=False,
        comment="消息内容"
    )
    
    # 消息元数据 (JSON格式)
    message_metadata: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="消息元数据(JSON格式)，包含来源、上下文等信息"
    )
    
    # 开始消费时间
    consumed_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="开始处理时间"
    )
    
    # 完成时间
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="处理完成时间"
    )
    
    # 错误信息
    error_message: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="处理失败时的错误信息"
    )
    
    # 过期时间
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="消息过期时间"
    )
    
    # 消息类型
    message_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="general",
        comment="消息类型(sensor_data/user_input/system_alert/general)"
    )
    
    # 优先级 (0-10, 数字越大优先级越高)
    priority: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        comment="消息优先级(0-10，数字越大优先级越高)"
    )
    
    # 消息状态
    status: Mapped[str] = mapped_column(
        SQLEnum('pending', 'processing', 'completed', 'failed', 'expired', name='message_status_enum'),
        default='pending',
        comment="消息状态：pending-待处理, processing-处理中, completed-已完成, failed-处理失败, expired-已过期"
    )
    
    # 接收时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        init=False,
        comment="消息接收时间"
    )
    
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
        init=False,
        comment="更新时间"
    )
    
    # 重试次数
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="重试次数"
    )
    
    # 最大重试次数
    max_retries: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
        comment="最大重试次数"
    )