import sys
import os

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from external_data_server.app_factory import app_context
from db_models.base import db
from db_models.chat_history import ChatHistory
from db_models.session import Session
from db_models.topic_memory import TopicMemory
from db_models.user import User

def setup_database():
    """
    在应用上下文中创建所有数据库表。
    如果表已存在，则不会重复创建。
    """
    print("正在进入应用上下文以创建或更新数据库表...")
    with app_context():
        db.create_all()
        print("数据库表已成功创建或更新！")

if __name__ == "__main__":
    setup_database()