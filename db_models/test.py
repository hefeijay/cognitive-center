import os

from dotenv import load_dotenv
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

# 从 .env 加载 DATABASE_URL
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cognitive_test.db")

# 创建对象的基类:
Base = declarative_base()

# 定义User对象:
class User(Base):
    # 表的名字:
    __tablename__ = 'usertst'

    # 表的结构:
    id = Column(String(20), primary_key=True)
    user_name = Column(String(20))

print("1")
# 初始化数据库连接（从环境变量读取，请在 .env 中配置 DATABASE_URL）
engine = create_engine(DATABASE_URL)
# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)

print("2")
Base.metadata.create_all(engine)

print("3")