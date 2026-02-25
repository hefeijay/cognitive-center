#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI决策消息数据库模型
用于存储和管理AI生成的决策消息、建议和分析结果
"""

from typing import Optional
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Index,
    String,
    TIMESTAMP,
    Integer,
    BIGINT,
    DECIMAL,
    Boolean,
    Enum as SQLEnum,
    text,
    Text as SQLText,
)
# 移除 MySQL 方言的 TEXT 导入，避免非 SQLite 兼容
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


class AIDecision(Base):
    """
    AI决策消息模型
    存储AI系统生成的各类决策消息、分析结果和操作建议
    """
    __tablename__ = "ai_decisions"
    __table_args__ = (
        Index("idx_ai_decisions_type", "type"),
        Index("idx_ai_decisions_status", "status"),
        Index("idx_ai_decisions_created_at", "created_at"),
        Index("idx_ai_decisions_priority", "priority"),
        Index("idx_ai_decisions_source_source_id", "source", "source_id"),
        Index("idx_ai_decisions_expires_at", "expires_at"),
    )
    
    # 主键ID
    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True, 
        init=False,
        comment="主键ID"
    )
    
    # 决策唯一标识
    decision_id: Mapped[str] = mapped_column(
        String(64), 
        unique=True, 
        nullable=False,
        comment="决策唯一标识"
    )
    
    # 消息类型
    type: Mapped[str] = mapped_column(
        SQLEnum('analysis', 'warning', 'action', 'optimization', name='decision_type_enum'),
        nullable=False,
        comment="消息类型：analysis-分析, warning-警告, action-操作, optimization-优化"
    )
    
    # 决策消息内容
    message: Mapped[str] = mapped_column(
        SQLText,
        nullable=False,
        comment="决策消息内容"
    )
    
    # 建议操作
    action: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="建议操作内容"
    )
    
    # 数据源类型
    source: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="数据源类型(sensor/device/system/manual)"
    )
    
    # 数据源ID
    source_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="数据源具体ID"
    )
    
    # 过期时间
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="过期时间，过期后消息将不再显示"
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        init=False,
        comment="创建时间"
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
    
    # 优先级 (0-10, 数字越大优先级越高)
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="优先级(0-10，数字越大优先级越高)"
    )

    # 置信度 (0-100)
    confidence: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(5, 2),
        default=Decimal('0.00'),
        comment="AI决策置信度(0-100)"
    )
    
    # 状态
    status: Mapped[str] = mapped_column(
        SQLEnum('active', 'processed', 'expired', name='decision_status_enum'),
        default='active',
        comment="状态：active-活跃, processed-已处理, expired-已过期"
    )


class MessageType(Base):
    """
    消息类型配置模型
    定义不同类型消息的显示样式和属性
    """
    __tablename__ = "message_types"
    
    # 主键ID
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        init=False,
        comment="主键ID"
    )
    
    # 消息类型
    type: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        comment="消息类型标识"
    )
    
    # 消息图标
    icon: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="消息图标(emoji)"
    )
    
    # 消息颜色
    color: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="消息颜色(hex或颜色名)"
    )
    
    # 类型描述
    description: Mapped[Optional[str]] = mapped_column(
        String(255),
        comment="消息类型描述"
    )
    
    # 是否启用
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="是否启用该消息类型"
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        init=False,
        comment="创建时间"
    )


class DecisionRule(Base):
    """
    AI决策规则模型
    定义触发AI决策的条件和规则
    """
    __tablename__ = "decision_rules"
    __table_args__ = (
        Index("idx_decision_rules_condition_type", "condition_type"),
        Index("idx_decision_rules_decision_type", "decision_type"),
        Index("idx_decision_rules_priority", "priority"),
        Index("idx_decision_rules_is_active", "is_active"),
    )
    
    # 主键ID
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        init=False,
        comment="主键ID"
    )
    
    # 规则唯一标识
    rule_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        comment="规则唯一标识"
    )
    
    # 规则名称
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="规则名称"
    )
    
    # 条件类型
    condition_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="条件类型(sensor/device/time/manual)"
    )
    
    # 条件配置 (JSON格式)
    condition_config: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="条件配置(JSON格式)"
    )
    
    # 消息模板
    message_template: Mapped[str] = mapped_column(
        SQLText,
        nullable=False,
        comment="消息模板，支持变量替换"
    )
    
    # 操作模板
    action_template: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="操作模板，支持变量替换"
    )
    
    # 决策类型
    decision_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="决策类型(analysis/warning/action/optimization)"
    )
    
    # 优先级
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="规则优先级(0-10)"
    )
    
    # 是否启用
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="是否启用该规则"
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        init=False,
        comment="创建时间"
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