# cognitive-model/agents/classification_agent.py

import logging
from typing import Dict, Any, Tuple, Optional
from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig, format_config_for_llm
from cognitive_model.config.prompt_manager import PromptManager

logger = logging.getLogger("cognitive_model.classification")

class ClassificationAgent:
    """
    记忆分类智能体 (ClassificationAgent)

    该智能体的核心职责是分析用户输入，并将其归纳为一个简短、精确、标准化的核心主题（Topic）。
    这个主题作为记忆索引的关键，用于在海马体（长期记忆系统）中高效地存储和检索相关信息。
    通过将每一轮对话都关联到一个明确的主题，系统能够更好地理解对话上下文，并在后续交互中
    快速找到相关的历史信息。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化分类智能体。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager

    async def run(self, user_input: str, model_config: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
            根据用户输入，调用大语言模型（LLM）生成一个简短、标准化的主题。

            业务逻辑:
            1.  **构建System Prompt**: 从 `prompt_manager` 获取为 `classification_agent` 定制的
                系统提示。这个提示会指导 LLM 如何进行主题分类，例如要求主题是名词短语、
                长度不超过N个词等。
            2.  **准备LLM输入**: 将构建好的系统提示和用户的原始输入 `user_input` 组合成
                一个符合 LLM API 要求的消息列表。
            3.  **调用LLM**: 使用 `execute_llm_call` 工具函数，以指定的配置（如模型 `gpt-4o`，
                温度 `0.6` 以获得较为一致但仍有一定创造性的输出）向 LLM 发起请求。
            4.  **清洗和格式化**: 对 LLM 返回的主题字符串进行处理，去除可能存在的多余空格、
                引号等，确保其格式的统一和干净。
            5.  **返回结果**: 返回处理后的主题字符串以及本次 LLM 调用的统计信息（如 Token 消耗）。

            Args:
                user_input (str): 用户的原始输入文本。

            Returns:
                Tuple[str, Dict[str, Any]]: 一个元组，包含两个元素：
                                        - 提取并格式化后的主题字符串。
                                        - LLM 调用的统计数据字典。
        """
        system_prompt = self.prompt_manager.format_prompt(
            agent_name="classification_agent",
            user_input=user_input
        )
        
        # 使用新的公共方法格式化配置
        config = format_config_for_llm(model_config)

        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        
        logger.info("调用分类智能体以获取主题...")
        config = LLMConfig(model="gpt-4o", temperature=0.6)
        topic, stats = await execute_llm_call(messages, config)
        topic = topic.strip().replace('"', '').replace("'", "")
        logger.info(f"分类智能体确定主题为: '{topic}'")
        return topic, stats