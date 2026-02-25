from typing import Optional, List
from datetime import datetime, timezone
import uuid

from sqlalchemy import (
    Index,
    String,
    TIMESTAMP,
    Integer,
    ForeignKey,
    Float,
    JSON,
    Text as SQLText,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


class Device(Base):
    """
    设备管理信息表
    用于管理各种类型的设备，包括传感器、执行器等
    """
    __tablename__ = "devices"
    __table_args__ = (
        Index("idx_devices_device_id", "device_id"),  # 设备唯一标识索引
        Index("idx_devices_status", "status"),  # 设备状态索引
        Index("idx_devices_ownership", "ownership"),  # 设备归属索引
        Index("idx_devices_type", "device_type_id"),  # 设备类型索引
    )

    # 主键ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)

    # 设备唯一标识符（UUID或序列号）
    device_id: Mapped[str] = mapped_column(
        String(128), 
        unique=True, 
        nullable=False,
        comment="设备唯一标识符 (UUID/SN)"
    )

    # 设备名称
    name: Mapped[str] = mapped_column(
        String(128), 
        nullable=False,
        comment="设备名称"
    )

    # 设备描述
    description: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="设备描述信息"
    )

    # 设备归属（用户、部门、项目等）
    ownership: Mapped[str] = mapped_column(
        String(128), 
        nullable=False,
        comment="设备归属"
    )

    # 设备类型ID（外键，关联设备类型表）
    device_type_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("device_types.id"),
        comment="设备类型ID"
    )

    # 设备型号/规格
    model: Mapped[Optional[str]] = mapped_column(
        String(128),
        comment="设备型号/规格"
    )

    # 制造商信息
    manufacturer: Mapped[Optional[str]] = mapped_column(
        String(128),
        comment="制造商"
    )

    # 设备序列号
    serial_number: Mapped[Optional[str]] = mapped_column(
        String(128),
        unique=True,
        comment="设备序列号"
    )

    # 设备安装位置描述
    location: Mapped[Optional[str]] = mapped_column(
        String(255),
        comment="设备安装位置"
    )

    # 关联养殖池ID（如果适用）
    pond_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ponds.id"),
        comment="所属养殖池ID"
    )

    # 固件版本
    firmware_version: Mapped[Optional[str]] = mapped_column(
        String(64),
        comment="固件版本"
    )

    # 硬件版本
    hardware_version: Mapped[Optional[str]] = mapped_column(
        String(64),
        comment="硬件版本"
    )

    # 设备IP地址
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # 支持IPv6
        comment="设备IP地址"
    )

    # MAC地址
    mac_address: Mapped[Optional[str]] = mapped_column(
        String(17),
        comment="MAC地址"
    )

    # 设备配置参数（JSON格式）
    config_json: Mapped[Optional[dict]] = mapped_column(
        JSON,
        comment="设备配置参数"
    )

    # 设备标签（便于分类和搜索）
    tags: Mapped[Optional[str]] = mapped_column(
        String(255),
        comment="设备标签，多个标签用逗号分隔"
    )

    # 安装时间
    installed_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="安装时间"
    )

    # 最后维护时间
    last_maintenance_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="最后维护时间"
    )

    # 下次维护时间
    next_maintenance_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="下次维护时间"
    )

    # 保修到期时间
    warranty_expires_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="保修到期时间"
    )

    # 设备退役时间（软删除）
    retired_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP,
        comment="设备退役时间"
    )

    # 记录更新时间
    updated_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), comment="最后更新时间", init=False)

    # 设备状态：active(活跃green), inactive(非活跃orange), maintenance(维护中blue), retired(已退役grey), error(故障red)
    status: Mapped[str] = mapped_column(
        String(50), 
        default="active", 
        nullable=False,
        comment="设备状态"
    )

    # 设备开关状态: on(开启), off(关闭), error(故障)
    switch_status: Mapped[str] = mapped_column(
        String(50), 
        default="off", 
        nullable=False,
        comment="开关状态"
    )

    # 设备优先级：high(高), medium(中), low(低)
    priority: Mapped[str] = mapped_column(
        String(20),
        default="medium",
        comment="设备优先级"
    )

    # 记录创建和更新时间
    created_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP, default=lambda: datetime.now(timezone.utc), comment="创建时间", init=False)
    
    # ORM 关系定义
    # device_type: Mapped[Optional["DeviceType"]] = relationship(back_populates="devices", init=False)
    # pond: Mapped[Optional["Pond"]] = relationship(back_populates="devices", init=False)


class DeviceType(Base):
    """
    设备类型表
    定义不同类型的设备分类
    """
    __tablename__ = "device_types"

    # 主键ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False)

    # 设备类型名称
    name: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        comment="设备类型名称"
    )

    # 设备类型描述
    description: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="设备类型描述"
    )

    # 设备类型代码
    type_code: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        comment="设备类型代码"
    )

    # 记录创建和更新时间
    created_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP, default=lambda: datetime.now(timezone.utc), comment="创建时间", init=False)
    updated_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), comment="最后更新时间", init=False)

    # ORM 关系定义
    # devices: Mapped[List["Device"]] = relationship(back_populates="device_type", cascade="all, delete-orphan", init=False)