# cognitive_model/config/prompt_manager.py

import logging
from typing import Dict, Any, Optional

from db_models.db_session import db_session_factory
from singa_one_server.services.prompt_service import prompt_service

logger = logging.getLogger("app.cognitive_model.config.prompt_manager")

class PromptManager:
    """
        提示管理器 (PromptManager) - 数据库驱动版

        一个用于从数据库加载、管理和格式化所有智能体提示（Prompts）的中央管理器。
        它的核心职责是实现 Prompt 的动态加载和集中管理，将 Prompt 内容与应用代码分离。

        核心设计思想:
        -   **数据库驱动 (Database-Driven)**: 所有 Prompt 模板都存储在数据库的 `prompts` 表中，
            取代了原先的 `prompts.json` 文件。这使得提示的更新可以动态进行，无需重启应用。
        -   **集中化管理 (Centralized Management)**: 提供一个单一的、全局的访问点 (`prompt_manager` 实例)
            来获取所有智能体的提示，确保了整个系统中 Prompt 的一致性。
        -   **动态格式化 (Dynamic Formatting)**: 支持在运行时将变量注入到提示模板中，
            以生成高度定制化的 Prompt。
        -   **鲁棒性与错误处理 (Robustness and Error Handling)**: 内置了对常见错误的优雅处理，
            例如数据库连接失败、格式化时缺少变量等。
        -   **结构化与可扩展性 (Structured and Extensible)**: 数据库结构支持一个智能体拥有多个
            不同用途的提示模板（通过 `template_key` 区分），易于扩展。

        业务逻辑:
        1.  **初始化 (`__init__`)**: 自动调用 `load_prompts` 方法从数据库加载所有提示。
        2.  **加载提示 (`load_prompts`)**: 通过 `prompt_service` 从数据库获取所有提示记录，
            并将其转换为与原 `prompts.json` 结构兼容的字典，存储在内存中。
        3.  **获取模板 (`get_prompt`)**: 根据智能体名称和可选的模板键，从内存中查找并返回原始的提示模板字符串。
        4.  **格式化提示 (`format_prompt`)**: 获取指定的模板，并将传入的关键字参数填充到模板的占位符中，
            生成最终的、可供 LLM 使用的完整提示。
    """
    def __init__(self):
        """
            初始化 PromptManager，并从数据库加载提示。
        """
        self._prompts: Dict[str, Any] = {}
        self.load_prompts()

    def load_prompts(self):
        """
            从数据库中加载所有提示模板。
        """
        try:
            with db_session_factory() as db:
                self._prompts = prompt_service.get_all_prompts_as_dict(db)
            logger.info(f"'智能体'加载成功!")
        except Exception as e:
            logger.exception(f"从数据库加载提示时发生未知错误: {e}")
            self._prompts = {}

    def get_prompt(self, agent_name: str, template_key: Optional[str] = None) -> Optional[str]:
        """
        获取指定智能体的原始提示模板。

        为了向后兼容和灵活性，此方法按以下顺序查找模板：
        1.  如果提供了 `template_key`，则在 `templates` 字典中精确查找该键。
        2.  如果 `template_key` 为 `None`：
            a. 首先尝试获取顶层的 `template` 键，以兼容旧的、单一模板格式。
            b. 如果找不到，则在 `templates` 字典中查找名为 `default` 的默认模板。

        Args:
            agent_name (str): 智能体的名称。
            template_key (Optional[str]): (可选) 要获取的模板的键。

        Returns:
            Optional[str]: 找到的提示模板字符串，如果未找到则返回 None。
        """
        agent_prompts = self._prompts.get(agent_name)
        if not agent_prompts:
            logger.warning(f"未找到名为 '{agent_name}' 的智能体提示配置。")
            return None

        # 场景 1: 明确指定了 template_key
        if template_key:
            template = agent_prompts.get("templates", {}).get(template_key)
            if not template:
                logger.warning(f"在 '{agent_name}' 的 templates 中未找到键为 '{template_key}' 的模板。")
            return template

        # 场景 2: 未指定 template_key，查找默认模板 (兼容新旧格式)
        template = agent_prompts.get("template") or agent_prompts.get("templates", {}).get("default")

        if not template:
            logger.warning(f"在 '{agent_name}' 中未找到默认模板 (既没有顶层 'template' 键, 也没有 'templates.default' 键)。")

        return template

    def format_prompt(self, agent_name: str, template_key: Optional[str] = None, **kwargs) -> str:
        """
            获取提示模板并使用提供的关键字参数进行格式化，生成最终的提示文本。

            这是与外部交互的主要方法。

            Args:
                agent_name (str): 智能体的名称。
                template_key (Optional[str]): (可选) 要使用的模板键。
                **kwargs: 用于填充模板中占位符（如 `{variable}`）的键值对。

            Returns:
                str: 格式化后的完整提示字符串。如果模板未找到或格式化失败，则返回一条错误信息。
        """
        template = self.get_prompt(agent_name, template_key)
        if not template:
            error_msg = f"无法为智能体 '{agent_name}' (键: {template_key}) 生成提示，因为模板未找到。"
            logger.error(error_msg)
            return error_msg 

        try:
            return template.format(**kwargs)
        except KeyError as e:
            error_msg = f"格式化提示 '{agent_name}' 时缺少变量: {e}"
            logger.error(error_msg)
            return error_msg

# prompt_manager = PromptManager()