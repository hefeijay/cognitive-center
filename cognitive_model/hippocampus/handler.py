# cognitive_model/hippocampus/handler.py

import logging
import asyncio
from typing import Dict, Any, List

# 导入相关的仓库和服务
from singa_one_server.repositories import topic_memory_repository
from singa_one_server.services import topic_memory_service

# 导入主题匹配智能体，这是实现主题聚类的核心依赖
from ..agents.topic_matching_agent import TopicMatcherAgent
from cognitive_model.config.prompt_manager import PromptManager

logger = logging.getLogger("app.cognitive_model.hippocampus.handler")

class HippocampusHandler:
    """
    管理认知中枢的长期记忆，这些记忆以“记忆节点”（Memory Node）的形式存储在数据库中。
    这个处理器模拟了大脑海马体的功能，负责新记忆的形成、巩固和检索。

    核心功能:
    - **主题聚类**: 利用 `TopicMatcherAgent` 解决“主题漂移”问题。当用户输入一个新的话题时，系统会判断这个新话题是否可以归入一个已有的、更宏观的主题下，而不是为每个微小的话题都创建一个新的记忆节点。
    - **记忆存储**: 将每个主主题（Master Topic）相关的对话历史、摘要、元数据等信息，以结构化的形式持久化到数据库中。
    - **记忆检索与管理**: 提供加载、创建和更新记忆节点的接口。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化海马体处理器。
        """
        # 实例化主题匹配智能体，用于后续的主题匹配决策
        self.topic_matcher = TopicMatcherAgent(prompt_manager)
        logger.info(f"'海马体'加载成功!")

    async def _get_all_existing_master_topics(self) -> List[str]:
        """
            从数据库中检索所有现有的主主题列表。
            这个列表将作为 `TopicMatcherAgent` 进行主题匹配时的候选集。

            Returns:
                List[str]: 一个包含所有已存在的主题字符串的列表。
        """
        try:
            loop = asyncio.get_running_loop()
            topic_objects = await loop.run_in_executor(
                None, topic_memory_repository.get_all_topic_memories
            )
            topics = [t.topic_name for t in topic_objects]
            logger.debug(f"从数据库发现的现有主主题列表: {topics}")
            return topics
        except Exception as e:
            logger.error(f"从数据库检索主主题时出错: {e}", exc_info=True)
            return []

    async def find_master_topic(self, candidate_topic: str) -> str:
        """
            为给定的候选主题，寻找或确定一个最终的主主题（Master Topic）。
            这是主题聚类流程的入口和核心，旨在将相似的对话归入同一个记忆节点。

            业务逻辑:
            1.  调用 `_get_all_existing_master_topics` 获取当前所有已存在的主题列表。
            2.  将候选主题和现有主题列表传递给 `TopicMatcherAgent`。
            3.  `TopicMatcherAgent` 内部会使用LLM来判断候选主题是否与某个现有主题足够相似。如果相似，则返回现有主题；如果不相似，则返回原始的候选主题，表示需要创建一个新的主主题。

            Args:
                candidate_topic (str): 由 `ClassificationAgent` 从用户输入中初步提取的主题。

            Returns:
                str: 最终确定的主主题字符串。
        """
        existing_topics = await self._get_all_existing_master_topics()
        master_topic, _ = await self.topic_matcher.match_topic(candidate_topic, existing_topics)
        return master_topic

    async def load_memory_node_by_master_topic(self, master_topic: str, session_id: str) -> Dict[str, Any]:
        """
            根据确定的主主题，从数据库加载一个现有的记忆节点或创建一个新的记忆节点。

            业务逻辑:
            1.  使用 `topic_memory_service.get_or_create_memory_node` 来处理业务逻辑。
            2.  服务层会检查主题是否存在。如果存在，则加载；如果不存在，则创建新的节点。
            3.  所有数据库操作都在服务和仓库层中异步执行。

            Args:
                master_topic (str): 经过匹配决策后的主主题。
                session_id (str): 当前会话的ID，用于关联记忆节点。

            Returns:
                Dict[str, Any]: 一个代表记忆节点内容的字典，无论是加载的还是新建的。
        """
        try:
            loop = asyncio.get_running_loop()
            memory_node_obj = await loop.run_in_executor(
                None,
                topic_memory_service.get_or_create_memory_node,
                master_topic,
                session_id
            )
            return memory_node_obj.to_dict() if memory_node_obj else None
        except Exception as e:
            logger.exception(f"加载或创建记忆节点 '{master_topic}' 时失败: {e}")
            # 在失败时返回一个安全的回退结构
            return {
                "topic": master_topic,
                "summary": "无法加载或创建记忆节点。",
                "history": [],
                "meta_data": {}
            }

    async def update_memory_node(self, memory_node_data: Dict[str, Any]):
        """
            将更新后的记忆节点数据写回到数据库中，实现记忆的持久化。

            业务逻辑:
            1.  调用 `topic_memory_repository.update_topic_memory` 来更新数据库中的记录。
            2.  此操作在执行器中异步运行。

            Args:
                memory_node_data (Dict[str, Any]): 包含要更新的记忆节点数据的字典。
        """
        try:
            loop = asyncio.get_running_loop()
            topic_id = memory_node_data.get("topic_id")
            if not topic_id:
                logger.error(f"更新记忆节点失败：memory_node_data 中缺少 'topic_id'。")
                return

            await loop.run_in_executor(
                None,
                topic_memory_repository.update_topic_memory,
                topic_id,
                memory_node_data
            )
            logger.info(f"记忆节点 '{memory_node_data.get('topic_name')}' 已成功更新到数据库。")
        except Exception as e:
            logger.exception(f"更新记忆节点 '{memory_node_data.get('topic_name')}' 到数据库时失败: {e}")