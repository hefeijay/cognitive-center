from typing import Optional, List
from datetime import datetime, timezone

from sqlalchemy import ( 
    Index, 
    String, 
    TIMESTAMP, 
    Integer, 
    ForeignKey, 
    Float,
    text,
) 
from sqlalchemy.orm import Mapped, mapped_column, relationship 
from .base import Base 

class Sensor(Base): 
    """ 
    物理传感器设备信息表 
    """ 
    __tablename__ = "sensors" 
    __table_args__ = ( 
        Index("idx_sensor_uuid", "sensor_id"), # 为传感器的唯一标识建立索引 
    ) 
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False) 
    
    # 传感器名称 
    name: Mapped[Optional[str]] = mapped_column(String(128), comment="传感器名称") 

    # 传感器的全局唯一标识符，可以是设备的序列号 
    sensor_id: Mapped[str] = mapped_column(String(128), unique=True, comment="传感器唯一标识符 (UUID/SN)") 
    
    # 外键，关联到养殖池 
    pond_id: Mapped[int] = mapped_column(ForeignKey("ponds.id"), comment="所属养殖池ID") 
    
    # 外键，关联到传感器类型 
    sensor_type_id: Mapped[int] = mapped_column(ForeignKey("sensor_types.id"), comment="传感器类型ID") 
    
    # 传感器型号 
    model: Mapped[Optional[str]] = mapped_column(String(128), comment="传感器型号") 

    # 安装日期 
    installed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, comment="安装日期") 
    
    # 传感器状态：active, inactive, maintenance 
    status: Mapped[str] = mapped_column(String(50), default="active", comment="传感器状态") 
    
    # 记录创建和更新时间 
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, 
        server_default=text("CURRENT_TIMESTAMP"), 
        nullable=False,
        init=False,
        comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, 
        server_default=text("CURRENT_TIMESTAMP"), 
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
        init=False,
        comment="最后更新时间"
    )

    # ORM 关系 
    pond: Mapped["Pond"] = relationship(back_populates="sensors", init=False) 
    sensor_type: Mapped["SensorType"] = relationship(back_populates="sensors", init=False) 
    readings: Mapped[list["SensorReading"]] = relationship(back_populates="sensor", cascade="all, delete-orphan", init=False)