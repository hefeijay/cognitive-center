#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头数据库模型
用于存储和管理摄像头状态、图片和健康检查数据
"""

from typing import Optional
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Index,
    String,
    TIMESTAMP,
    Integer,
    BIGINT,
    DECIMAL,
    Boolean,
    text,
)
# 移除未使用的 MySQL 方言 TEXT 导入以提升兼容性
# from sqlalchemy.dialects.mysql import TEXT
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class CameraStatus(Base):
    """
    摄像头状态模型
    存储摄像头的基本状态信息
    """
    __tablename__ = "camera_status"
    __table_args__ = (
        Index("idx_camera_status_camera_id", "camera_id"),
        Index("idx_camera_status_status", "status"),
        Index("idx_camera_status_location", "location"),
        Index("idx_camera_status_created_at", "created_at"),
    )
    
    # 主键ID
    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True, 
        comment="主键ID"
    )
    
    # 摄像头ID
    camera_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="摄像头ID"
    )
    
    # 摄像头名称
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="摄像头名称"
    )
    
    # 摄像头位置
    location: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="摄像头位置"
    )
    
    # 摄像头状态
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="摄像头状态(在线/离线)"
    )
    
    # 画质
    quality: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="画质(高/中/低)"
    )
    
    # 分辨率
    resolution: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="分辨率"
    )
    
    # 最后更新时间戳
    last_update: Mapped[int] = mapped_column(
        BIGINT,
        nullable=False,
        comment="最后更新时间戳(毫秒)"
    )
    
    # 最后更新时间字符串
    last_update_time: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="最后更新时间字符串"
    )
    
    # 温度 - 可选字段，添加默认值
    temperature: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(5, 2),
        default=None,
        comment="温度"
    )
    
    # 以下字段有默认值，放在最后
    # 帧率
    fps: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="帧率"
    )
    
    # 连通性
    connectivity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="连通性百分比"
    )
    
    # 是否录制
    recording: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否正在录制"
    )
    
    # 夜视功能
    night_vision: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否开启夜视功能"
    )
    
    # 运动检测
    motion_detection: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="是否开启运动检测"
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        default=None,
        comment="创建时间"
    )
    
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
        default=None,
        comment="更新时间"
    )


class CameraImage(Base):
    """
    摄像头图片模型
    存储摄像头拍摄的图片信息
    """
    __tablename__ = "camera_images"
    __table_args__ = (
        Index("idx_camera_images_camera_id", "camera_id"),
        Index("idx_camera_images_status", "status"),
        Index("idx_camera_images_timestamp", "timestamp"),
        Index("idx_camera_images_created_at", "created_at"),
    )
    
    # 主键ID
    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True, 
        comment="主键ID"
    )
    
    # 摄像头ID
    camera_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="摄像头ID"
    )
    
    # 摄像头名称
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="摄像头名称"
    )
    
    # 摄像头位置
    location: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="摄像头位置"
    )
    
    # 摄像头状态
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="摄像头状态(在线/离线)"
    )
    
    # 图片URL
    image_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="图片URL"
    )
    
    # 最后更新时间戳
    last_update: Mapped[int] = mapped_column(
        BIGINT,
        nullable=False,
        comment="最后更新时间戳(毫秒)"
    )
    
    # 时间戳
    timestamp: Mapped[int] = mapped_column(
        BIGINT,
        nullable=False,
        comment="图片时间戳(毫秒)"
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        default=None,
        comment="创建时间"
    )
    
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
        default=None,
        comment="更新时间"
    )
    
    # 时间戳字符串
    timestamp_str: Mapped[str] = mapped_column(
        String(50),
        default="",
        comment="图片时间戳字符串"
    )
    
    # 图片宽度
    width: Mapped[int] = mapped_column(
        Integer,
        default=1920,
        comment="图片宽度"
    )
    
    # 图片高度
    height: Mapped[int] = mapped_column(
        Integer,
        default=1080,
        comment="图片高度"
    )
    
    # 图片格式
    format: Mapped[str] = mapped_column(
        String(20),
        default="jpg",
        comment="图片格式(jpg/png等)"
    )
    
    # 图片大小
    size: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="图片大小(字节)"
    )
    
    # 帧率
    fps: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="帧率"
    )


class CameraHealth(Base):
    """
    摄像头健康检查模型
    存储摄像头的健康状态和检查结果
    """
    __tablename__ = "camera_health"
    __table_args__ = (
        Index("idx_camera_health_camera_id", "camera_id"),
        Index("idx_camera_health_health_status", "health_status"),
        Index("idx_camera_health_overall_score", "overall_score"),
        Index("idx_camera_health_created_at", "created_at"),
    )
    
    # 主键ID
    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True, 
        comment="主键ID"
    )
    
    # 摄像头ID
    camera_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="摄像头ID"
    )
    
    # 摄像头名称
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="摄像头名称"
    )
    
    # 摄像头位置
    location: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="摄像头位置"
    )
    
    # 健康状态
    health_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="健康状态(优秀/良好/一般/需要维护/离线)"
    )
    
    # 总体健康分数
    overall_score: Mapped[Decimal] = mapped_column(
        DECIMAL(5, 2),
        nullable=False,
        comment="总体健康分数(0-100)"
    )
    
    # 连通性检查状态
    connectivity_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="连通性检查状态"
    )
    
    # 连通性分数
    connectivity_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="连通性分数"
    )
    
    # 连通性消息
    connectivity_message: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="连通性检查消息"
    )
    
    # 图像质量检查状态
    image_quality_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="图像质量检查状态"
    )
    
    # 图像质量分数
    image_quality_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="图像质量分数"
    )
    
    # 图像质量消息
    image_quality_message: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="图像质量检查消息"
    )
    
    # 硬件检查状态
    hardware_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="硬件检查状态"
    )
    
    # 硬件分数
    hardware_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="硬件分数"
    )
    
    # 硬件消息
    hardware_message: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="硬件检查消息"
    )
    
    # 存储检查状态
    storage_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="存储检查状态"
    )
    
    # 存储分数
    storage_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="存储分数"
    )
    
    # 存储检查消息
    storage_message: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="存储检查消息"
    )
    
    # 检查时间戳
    timestamp: Mapped[int] = mapped_column(
        BIGINT,
        nullable=False,
        comment="检查时间戳(毫秒)"
    )
    
    # 最后检查时间
    last_check: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="最后检查时间字符串"
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        comment="创建时间"
    )
    
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
        comment="更新时间"
    )
    
    # 温度
    temperature: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(5, 2),
        default=None,
        comment="温度"
    )
    
    # 运行时间
    uptime_hours: Mapped[Optional[int]] = mapped_column(
        Integer,
        default=None,
        comment="运行时间(小时)"
    )