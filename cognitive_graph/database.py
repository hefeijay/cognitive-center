#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库操作模块
提供数据库访问和操作功能
"""

import sys
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

# 添加项目根目录到Python路径
sys.path.append('/home/gmm/srv/cognitive-center')
# 使用 japan-aquaculture-project 的数据库模型
sys.path.append('/home/gmm/srv/japan-aquaculture-project/backend')

from db_models.message_queue import MessageQueue
from db_models.ai_decision import AIDecision
from db_models.chat_history import ChatHistory
from db_models.prompt import Prompt
from .config import get_db_session, close_db_session


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.session: Optional[Session] = None
    
    def __enter__(self):
        self.session = get_db_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            close_db_session(self.session)
    
    # 消息队列操作
    def get_pending_messages(self, limit: int = 10) -> List[MessageQueue]:
        """获取待处理的消息"""
        return (self.session.query(MessageQueue)
                .filter(MessageQueue.status == 'pending')
                .order_by(desc(MessageQueue.priority), MessageQueue.created_at)
                .limit(limit)
                .all())
    
    def get_processing_messages(self, limit: int = 10) -> List[MessageQueue]:
        """获取正在处理的消息"""
        return (self.session.query(MessageQueue)
                .filter(MessageQueue.status == 'processing')
                .order_by(desc(MessageQueue.priority), MessageQueue.created_at)
                .limit(limit)
                .all())
    
    def get_message_by_id(self, message_id: str) -> Optional[MessageQueue]:
        """根据消息ID获取消息"""
        return (self.session.query(MessageQueue)
                .filter(MessageQueue.message_id == message_id)
                .first())
    
    def update_message_status(self, message_id: str, status: str, 
                            consumed_at: Optional[datetime] = None,
                            completed_at: Optional[datetime] = None,
                            error_message: Optional[str] = None) -> bool:
        """更新消息状态"""
        try:
            message = (self.session.query(MessageQueue)
                      .filter(MessageQueue.message_id == message_id)
                      .first())
            if message:
                message.status = status
                if consumed_at:
                    message.consumed_at = consumed_at
                if completed_at:
                    message.completed_at = completed_at
                if error_message:
                    message.error_message = error_message
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            raise e
    
    def create_message(self, message_id: str, content: str, 
                      message_type: str = "general", 
                      metadata: Optional[str] = None,
                      priority: int = 5) -> MessageQueue:
        """创建新消息"""
        try:
            message = MessageQueue(
                message_id=message_id,
                content=content,
                message_metadata=metadata,
                consumed_at=None,
                completed_at=None,
                error_message=None,
                expires_at=None,
                message_type=message_type,
                priority=priority,
                status='pending'
            )
            self.session.add(message)
            self.session.commit()
            return message
        except Exception as e:
            self.session.rollback()
            raise e
    
    # AI决策操作
    def create_ai_decision(self, decision_id: str, decision_type: str, message: str, 
                          confidence: float = 0.0, action: str = None, 
                          source: str = None, source_id: str = None, 
                          priority: int = 0, expires_at: datetime = None) -> AIDecision:
        """
        创建AI决策记录
        
        Args:
            decision_id: 决策唯一标识
            decision_type: 决策类型 (analysis, warning, action, optimization)
            message: 决策消息内容
            confidence: 置信度 (0-100)
            action: 建议操作内容
            source: 数据源类型
            source_id: 数据源ID
            priority: 优先级 (0-10)
            expires_at: 过期时间
            
        Returns:
            AIDecision: 创建的AI决策对象
        """
        try:
            decision = AIDecision(
                decision_id=decision_id,
                type=decision_type,
                message=message,
                action=action,
                source=source,
                source_id=source_id,
                priority=priority,
                confidence=Decimal(str(confidence)),
                status='active',
                expires_at=expires_at
            )
            
            self.session.add(decision)
            self.session.commit()
            return decision
            
        except Exception as e:
            self.session.rollback()
            raise e
    
    # 对话历史操作
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[ChatHistory]:
        """获取对话历史"""
        return (self.session.query(ChatHistory)
                .filter(and_(
                    ChatHistory.session_id == session_id,
                    ChatHistory.status == 'active'
                ))
                .order_by(ChatHistory.timestamp)
                .limit(limit)
                .all())
    
    def save_chat_message(self, session_id: str, role: str, content: str,
                         message_type: str = "text", 
                         message_id: Optional[str] = None,
                         tool_calls: Optional[str] = None,
                         meta_data: Optional[str] = None) -> ChatHistory:
        """保存对话消息"""
        try:
            chat_message = ChatHistory(
                session_id=session_id,
                role=role,
                content=content,
                type=message_type,
                message_id=message_id,
                tool_calls=tool_calls,
                meta_data=meta_data,
                status='active'
            )
            self.session.add(chat_message)
            self.session.commit()
            return chat_message
        except Exception as e:
            self.session.rollback()
            raise e
    
    # Prompt操作
    def get_agent_prompt(self, agent_name: str, template_key: str = "graph_agent") -> Optional[Prompt]:
        """获取智能体的prompt模板"""
        return (self.session.query(Prompt)
                .filter(and_(
                    Prompt.agent_name == agent_name,
                    Prompt.template_key == template_key
                ))
                .first())
    
    def get_all_agent_prompts(self, template_key: str = "graph_agent") -> List[Prompt]:
        """获取所有智能体的prompt模板"""
        return (self.session.query(Prompt)
                .filter(Prompt.template_key == template_key)
                .all())