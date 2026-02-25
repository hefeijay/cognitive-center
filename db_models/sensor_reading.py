from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    Float,
    TIMESTAMP,
    ForeignKey,
    Index,
    String,
    Text,
) 
from sqlalchemy.orm import relationship, Mapped, mapped_column 
 
from .base import Base 
 
 
class SensorReading(Base):
    """
    传感器读数记录表（核心数据表）

    功能说明：
    - 存储单次传感器采集的原始读数及时间戳。
    - 新增快照字段：type_name、description、unit，来源于传感器类型表，在插入/更新时由数据库触发器自动回填。

    字段说明：
    - sensor_id (int): 关联到 sensors.id，表示具体的传感器设备。
    - value (float): 采集到的数值。
    - recorded_at (timestamp): 该数值的记录时间。
    - type_name (varchar(128), 可空): 传感器类型名称快照（来自 sensor_types.type_name）。
    - description (text, 可空): 传感器类型描述快照（来自 sensor_types.description）。
    - unit (varchar(50), 可空): 数值单位快照（来自 sensor_types.unit）。

    使用示例：
    - 查询最近一条读数及其单位：
      SELECT value, unit FROM sensor_readings WHERE sensor_id = :sid ORDER BY recorded_at DESC LIMIT 1;
    """
    __tablename__ = "sensor_readings" 
    __table_args__ = ( 
        # 为最常见的查询（查询某个传感器在一段时间内的数据）建立复合索引 
        Index("idx_sensor_id_recorded_at", "sensor_id", "recorded_at"), 
    ) 
     
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False) 
     
    # 外键，关联到具体的传感器设备 
    sensor_id: Mapped[int] = mapped_column(ForeignKey("sensors.id"), comment="传感器设备ID") 
     
    # 记录的数值，使用浮点数以兼容各种类型的数据 
    value: Mapped[float] = mapped_column(Float, comment="传感器读数值") 
     
    # 数据记录的时间戳 
    recorded_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP, default=datetime.now(timezone.utc), comment="数据记录时间") 

    # 直观快照字段（与 sensor_types 关联信息的写时快照）
    # 这些快照字段由数据库触发器在插入/更新时自动回填，不需要参与 dataclass 的 __init__ 参数
    type_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, init=False, comment="传感器类型名称快照")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, init=False, comment="传感器类型描述快照")
    unit: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, init=False, comment="传感器读数单位快照")
 
    # ORM 关系 
    sensor: Mapped["Sensor"] = relationship(back_populates="readings", init=False)
