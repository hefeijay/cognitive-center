# db_models/db_session.py

import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import AppConfig

logger = logging.getLogger(__name__)

_engine = None
_Session = None

def get_engine():
    """
    获取数据库引擎的单例实例。

    如果引擎尚不存在，则根据 AppConfig 中的配置创建一个新的数据库引擎。
    使用单例模式确保整个应用只使用一个引擎实例，从而有效地管理数据库连接池。

    Returns:
        Engine: SQLAlchemy 数据库引擎实例。
    """
    global _engine
    if _engine is None:
        config = AppConfig()
        # logger.info(f"为数据库 {config.DATABASE_URL} 初始化数据库引擎")
        _engine = create_engine(
            config.DATABASE_URL,
            pool_size=15,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=600,
            pool_pre_ping=True,
            pool_use_lifo=True,
        )
    return _engine

def get_session_factory():
    """
    获取数据库会話工厂的单例实例。

    如果会话工厂不存在，则使用 get_engine() 获取的引擎创建一个新的会话工厂。
    这确保了所有的会话都绑定到同一个引擎上。

    Returns:
        sessionmaker: SQLAlchemy 会话工厂。
    """
    global _Session
    if _Session is None:
        engine = get_engine()
        _Session = sessionmaker(bind=engine)
    return _Session

@contextmanager
def db_session_factory():
    """
    提供一个围绕一系列数据库操作的事务作用域。

    这是一个上下文管理器，它从会话工厂获取一个新的数据库会话，
    并在 `with` 块内提供该会话。如果代码块成功执行，则提交事务；
    如果发生异常，则回滚事务。无论成功与否，会话最终都会被关闭。

    Usage:
        with db_session_factory() as session:
            # 在这里执行数据库操作
            session.add(some_object)
            session.commit()
    """
    session_factory = get_session_factory()
    session = session_factory()
    logger.debug("数据库会话已创建。")
    try:
        yield session
        # 注意：在仓库层，通常会在每个函数结束时单独调用 commit()，
        # 这里的 commit() 作为一种保障，但最佳实践是在调用方控制事务。
    except Exception:
        logger.error("数据库会话因异常而回滚。", exc_info=True)
        session.rollback()
        raise
    finally:
        session.close()
        logger.debug("数据库会话已关闭。")