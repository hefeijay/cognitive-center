from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import (
    Index,
    String,
    TIMESTAMP,
    Integer,
    Text as SQLText,
)
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class TopicMemory(Base):
    """
    话题记忆表 - 用于存储海马体式的分类记忆用户聊天话题主题
    
    该表用于记录和管理用户对话中的主题分类，实现类似海马体的记忆功能，
    能够对聊天内容进行主题归类、总结和检索。
    """
    __tablename__ = "topic_memory"
    __table_args__ = (
        Index("idx_topic_memory_topic_id", "topic_id"),
        Index("idx_topic_memory_session_id", "session_id"),
        Index("idx_topic_memory_create_time", "create_time"),
        Index("idx_topic_memory_update_time", "update_time"),
    )
    
    # 自增主键ID
    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True, 
        init=False,
        comment="自增主键ID"
    )
    
    # 主题唯一标识ID
    topic_id: Mapped[str] = mapped_column(
        String(128), 
        unique=True,
        comment="主题唯一标识ID，用于记录不同的话题"
    )
    
    # 主题名称
    topic_name: Mapped[str] = mapped_column(
        String(255),
        comment="归类的主题名称"
    )
    
    # 主题内容总结
    topic_content: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="总结的主题内容描述"
    )
        
    # 关联标签
    related_tags: Mapped[Optional[str]] = mapped_column(
        SQLText,
        comment="该主题的关键标签，多个标签用逗号分隔"
    )
    
    # 创建来源会话ID
    session_id: Mapped[str] = mapped_column(
        String(128),
        comment="创建该主题的会话ID"
    )
    
    # 创建时间
    create_time: Mapped[Optional[TIMESTAMP]] = mapped_column(
        TIMESTAMP,
        default=datetime.now(timezone.utc),
        comment="主题最初创建时间"
    )
    
    # 更新时间
    update_time: Mapped[Optional[TIMESTAMP]] = mapped_column(
        TIMESTAMP,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        comment="主题最新更新时间"
    )

    # 状态字段（可选，用于软删除等操作）
    status: Mapped[Optional[str]] = mapped_column(
        String(50),
        default="active",
        comment="主题状态：active-活跃，inactive-非活跃，deleted-已删除"
    )
    
    def __repr__(self) -> str:
        """
        返回对象的字符串表示
        
        Returns:
            str: 包含主要字段信息的字符串
        """
        return f"<TopicMemory(topic_id='{self.topic_id}', topic_name='{self.topic_name}', session_id='{self.session_id}')>"
    
    def to_dict(self) -> dict:
        """
        将对象转换为字典格式
        
        Returns:
            dict: 包含所有字段的字典
        """
        return {
            'id': self.id,
            'topic_id': self.topic_id,
            'topic_name': self.topic_name,
            'topic_content': self.topic_content,
            'create_time': self.create_time.isoformat() if self.create_time else None,
            'update_time': self.update_time.isoformat() if self.update_time else None,
            'related_tags': self.related_tags,
            'session_id': self.session_id,
            'status': self.status
        }