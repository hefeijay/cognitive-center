#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知图数据库配置模块
从环境变量中读取数据库连接信息和其他配置
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class CognitiveGraphConfig(BaseSettings):
    """认知图配置类"""
    
    # 数据库配置
    DATABASE_URL: str = "sqlite:///cognitive_graph.db"
    
    # OpenAI API配置 - 从.env文件读取并清理特殊字符
    OPENAI_API_KEY: str
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 清理API密钥中的特殊字符
        if self.OPENAI_API_KEY:
            self.OPENAI_API_KEY = self.OPENAI_API_KEY.replace('\r', '').replace('\n', '').strip()
    
    # 项目路径配置
    PROJECT_ROOT_PATH: str = "/usr/henry/cognitive-center"
    COGNITIVE_MODEL_PATH: str = "/usr/henry/cognitive-center/cognitive_model"
    
    # MCP服务配置
    MCP_SERVER_URL: Optional[str] = "http://localhost:8080"
    
    # 多智能体配置
    MAX_CONVERSATION_ROUNDS: int = 10
    MESSAGE_PROCESSING_TIMEOUT: int = 300  # 5分钟
    
    # 消息队列配置
    MESSAGE_BATCH_SIZE: int = 10
    MESSAGE_POLL_INTERVAL: int = 5  # 秒
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# 全局配置实例
config = CognitiveGraphConfig()

# 数据库引擎和会话
engine = create_engine(
    config.DATABASE_URL,
    echo=False,  # 设置为True可以看到SQL语句
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """获取数据库会话"""
    session = SessionLocal()
    try:
        return session
    except Exception:
        session.close()
        raise


def close_db_session(session):
    """关闭数据库会话"""
    try:
        session.close()
    except Exception:
        pass


# 设置OpenAI API Key
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
print(f"秘钥：{os.environ['OPENAI_API_KEY']}")