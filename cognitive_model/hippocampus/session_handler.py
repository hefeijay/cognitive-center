# cognitive_model/hippocampus/session_handler.py

import logging
import asyncio
import json

# Import the repository to interact with the database
from singa_one_server.repositories import chat_history_repository
from db_models.model import ChatHistory

logger = logging.getLogger(__name__)

def _db_history_to_dict(history_entry: ChatHistory) -> dict:
    """Converts a ChatHistory ORM object to a dictionary."""
    # The application expects tool_calls and metadata to be objects, not JSON strings.
    tool_calls = None
    if history_entry.tool_calls:
        try:
            tool_calls = json.loads(history_entry.tool_calls)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode tool_calls JSON for message {history_entry.message_id}")
            tool_calls = None

    meta_data = None
    if history_entry.meta_data:
        try:
            meta_data = json.loads(history_entry.meta_data)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode meta_data JSON for message {history_entry.message_id}")
            meta_data = None

    return {
        "role": history_entry.role,
        "content": history_entry.content,
        "type": history_entry.type,
        "timestamp": history_entry.timestamp.timestamp(), # Convert datetime back to unix timestamp
        "message_id": history_entry.message_id,
        "tool_calls": tool_calls,
        "meta_data": meta_data,
    }

async def load_session_history(session_id: str) -> list:
    """
    异步加载指定会话ID的完整对话历史。
    这模拟了短期记忆的快速检索过程。

    业务逻辑:
    1.  调用 `chat_history_repository.get_history_by_session_id` 从数据库获取历史记录。
    2.  由于仓库函数是同步的，因此在执行器中运行以避免阻塞asyncio事件循环。
    3.  将每个数据库ORM对象转换为应用程序期望的字典格式。
    4.  处理数据库访问期间可能发生的异常。

    Args:
        session_id (str): 要加载历史记录的会话ID。

    Returns:
        list: 一个包含该会话所有对话轮次（turn）的列表。每个轮次是一个字典。
    """
    try:
        loop = asyncio.get_running_loop()
        # 为同步数据库调用使用run_in_executor
        db_history = await loop.run_in_executor(
            None, chat_history_repository.get_history_by_session_id, session_id
        )
        
        history = [_db_history_to_dict(entry) for entry in db_history]
        
        logger.debug(f"已从数据库为会话 {session_id} 加载 {len(history)} 轮对话。")
        return history
    except Exception as e:
        logger.exception(f"为会话 {session_id} 从数据库加载历史记录时发生意外错误: {e}")
        return []

async def save_session_turn(session_id: str, turn_data: dict):
    """
    异步地将一轮新的对话数据保存到指定会话在数据库中的历史记录。

    业务逻辑:
    1.  输入的 `turn_data` 字典被传递给 `chat_history_repository.add_message_to_history` 函数。
    2.  仓库负责将其转换为 `ChatHistory` ORM对象并提交到数据库。
    3.  同步的仓库函数在执行器中运行，以保持操作的非阻塞性。

    Args:
        session_id (str): 目标会话的ID。
        turn_data (dict): 代表一轮对话的字典，必须包含 'role', 'content' 等。
                          它也应该包含 'session_id'。
    """
    # 确保 turn_data 中有 session_id，因为仓库函数需要它。
    if 'session_id' not in turn_data:
        turn_data['session_id'] = session_id

    try:
        loop = asyncio.get_running_loop()
        # 为同步数据库调用使用run_in_executor
        await loop.run_in_executor(
            None, chat_history_repository.add_message_to_history, turn_data
        )
        logger.debug(f"已为会话 {session_id} 向数据库保存新的一轮对话。")
    except Exception as e:
        logger.exception(f"为会话 {session_id} 保存一轮对话至数据库时发生意外错误: {e}")

async def clear_session_history(session_id: str):
    """
    异步地清除指定会话ID的历史记录。
    注意：对于数据库持久化，'清除'可能意味着将记录标记为非活动状态，
    而不是删除它们。此实现是一个占位符，可能需要根据
    最终的数据管理策略进行调整。目前，它不执行任何操作。

    Args:
        session_id (str): 要清除历史记录的会话ID。
    """
    logger.warning(f"为会话 {session_id} 调用了 clear_session_history，但该功能未针对数据库存储实现。未执行任何操作。")
    # 要实现此功能，您需要一个类似以下的仓库函数：
    # chat_history_repository.delete_history_by_session_id(session_id)
    # 或
    # chat_history_repository.deactivate_history_by_session_id(session_id)
    await asyncio.sleep(0) # 使其可等待