# cognitive_model/tasks/task_handler.py

import uuid
import logging
from typing import Dict, Any, List, Optional

from db_models.task import Task
from singa_one_server.repositories.task_repository import TaskRepository

logger = logging.getLogger("app.cognitive_model.tasks.task_handler")


class TaskHandler:
    """
    管理和操作与特定会话主题（Topic）关联的任务（Task）。
    在认知模型中，一个“任务”通常代表一个需要执行的工具调用（Tool Use），这个处理器负责跟踪这些任务的生命周期。
    """

    def __init__(self):
        """
        初始化任务处理器。
        """
        self.task_repository = TaskRepository()
        logger.info("'任务中心'加载成功!")

    def create_task(self, topic_normalized: str, tool_name: str, tool_args: Any, mode: str) -> Task:
        """
        为一个主题创建一个新的任务，并将其存储到数据库中。

        Args:
            topic_normalized (str): 标准化后的会话主题。
            tool_name (str): 任务所使用的工具的名称。
            tool_args (Any): 传递给工具的参数。
            mode (str): 任务的执行模式，通常是 'sync' (同步) 或 'async' (异步)。

        Returns:
            Task: 已创建的新任务的数据库对象。
        """
        task_id = str(uuid.uuid4())
        request_payload = {"args": tool_args}

        db_task = self.task_repository.create_task(
            task_id=task_id,
            topic=topic_normalized,
            tool_name=tool_name,
            mode=mode,
            request=request_payload
        )

        logger.warning(f"为主题 '{topic_normalized}' 创建了新的 {mode} 任务 {task_id}")
        return db_task

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        按ID从数据库中获取任务。

        Args:
            task_id (str): 任务的唯一ID。

        Returns:
            Optional[Task]: 找到的任务对象，否则为None。
        """
        return self.task_repository.get_task_by_id(task_id=task_id)

    def get_tasks_by_topic(self, topic_normalized: str) -> List[Task]:
        """
        按主题从数据库中获取所有任务。

        Args:
            topic_normalized (str): 标准化后的会话主题。

        Returns:
            List[Task]: 属于该主题的任务列表。
        """
        return self.task_repository.get_tasks_by_topic(topic=topic_normalized)

    def update_task_status(self, task_id: str, new_status: str, response: Any = None) -> Optional[Task]:
        """
        更新一个特定任务的状态及其执行结果。

        Args:
            task_id (str): 要更新的任务的唯一ID。
            new_status (str): 新的任务状态 (例如, 'running', 'completed', 'failed')。
            response (Any, optional): 任务的执行结果。默认为 None。

        Returns:
            Optional[Task]: 更新后的任务对象，如果未找到则为None。
        """
        db_task = self.task_repository.update_task(
            task_id=task_id,
            status=new_status,
            response=response
        )

        if db_task:
            logger.debug(f"任务 {task_id} 的状态已更新为 '{new_status}'。")
        else:
            logger.error(f"尝试更新任务失败：未找到任务ID '{task_id}'。")

        return db_task