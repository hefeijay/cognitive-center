#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库表创建脚本
创建认知图系统所需的所有数据库表
"""

import os
import sys
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append('/home/gmm/srv/cognitive-center')
# 使用 japan-aquaculture-project 的数据库模型
sys.path.append('/home/gmm/srv/japan-aquaculture-project/backend')

# 加载环境变量
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

from db_models.base import Base
from db_models.message_queue import MessageQueue
from db_models.ai_decision import AIDecision
from db_models.chat_history import ChatHistory
from db_models.prompt import Prompt


def create_tables():
    """
    创建所有数据库表
    注意：如果数据库表已存在（通过 japan-aquaculture-project 创建），则会跳过
    """
    try:
        print("开始检查/创建数据库表...")
        
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
        
        # 创建所有表（如果表已存在则跳过）
        print("\n正在检查并创建必需的表...")
        Base.metadata.create_all(bind=engine)
        
        print("\n✅ 数据库表检查完成！")
        print("\n📋 Cognitive Center 所需的表:")
        print("  - message_queue: 消息队列表")
        print("  - ai_decisions: AI决策表")
        print("  - chat_history: 聊天历史表")
        print("  - prompts: 智能体提示词表")
        print("\n💡 注意：如果这些表已在 japan-aquaculture-project 中创建，")
        print("   则会自动跳过创建，直接使用已有的表。")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据库操作失败: {e}")
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