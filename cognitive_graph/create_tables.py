#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库表创建脚本
创建认知图系统所需的所有数据库表
"""

import os
import sys
from sqlalchemy import create_engine

# 添加项目根目录到Python路径
sys.path.append('/usr/henry/cognitive-center')

from db_models.base import Base
from db_models.message_queue import MessageQueue
from db_models.ai_decision import AIDecision
from db_models.chat_history import ChatHistory
from db_models.prompt import Prompt


def create_tables():
    """创建所有数据库表"""
    try:
        print("开始创建数据库表...")
        
        # 使用SQLite数据库进行测试
        database_url = os.getenv("DATABASE_URL", "sqlite:///./cognitive_graph.db")
        print(f"数据库URL: {database_url}")
        
        # 创建数据库引擎
        if "sqlite" in database_url:
            # SQLite配置
            engine = create_engine(database_url, echo=False)
        else:
            # MySQL配置
            engine = create_engine(
                database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        
        print("数据库表创建成功！")
        print("已创建的表:")
        print("- message_queue: 消息队列表")
        print("- ai_decision: AI决策表")
        print("- chat_history: 聊天历史表")
        print("- prompt: 智能体提示词表")
        
        return True
        
    except Exception as e:
        print(f"创建数据库表失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_tables()
    if not success:
        print("\n数据库表创建失败，请检查配置和网络连接")
        sys.exit(1)
    else:
        print("\n数据库初始化完成！")