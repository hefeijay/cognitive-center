from typing import Optional 
from datetime import datetime, timezone 

from sqlalchemy import ( 
    Index, 
    String, 
    TIMESTAMP, 
    Integer, 
    ForeignKey, 
    Float, 
    Text as SQLText,
) 
from sqlalchemy.orm import Mapped, mapped_column, relationship 
from .base import Base 

class Pond(Base): 
    """ 
    养殖池信息表 
    """ 
    __tablename__ = "ponds" 
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, init=False) 
    
    # 养殖池的唯一标识或名称，方便人类阅读 
    name: Mapped[str] = mapped_column(String(128), unique=True, comment="养殖池名称") 
    
    # 养殖池的地理位置或其他描述信息 
    location: Mapped[Optional[str]] = mapped_column(String(255), comment="位置信息") 
    
    # 备注信息 
    description: Mapped[Optional[str]] = mapped_column(SQLText, comment="描述/备注") 
    
    # 记录创建和更新时间 
    created_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP, default=datetime.now(timezone.utc), comment="创建时间")
    updated_at: Mapped[Optional[TIMESTAMP]] = mapped_column(TIMESTAMP, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment="最后更新时间")

    # ORM 关系：一个养殖池可以有多个传感器 
    sensors: Mapped[list["Sensor"]] = relationship(back_populates="pond", cascade="all, delete-orphan", init=False)