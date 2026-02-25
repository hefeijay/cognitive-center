# cognitive_model/agents/topic_matching_agent.py

import logging
from typing import Dict, Any, Tuple, List
from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig
from cognitive_model.config.prompt_manager import PromptManager

logger = logging.getLogger("app.cognitive_model.topic")

class TopicMatcherAgent:
    """
    主题匹配智能体 (TopicMatcherAgent)

    该智能体负责将一个新的候选主题与一个已有的主题列表进行比较和匹配。
    其核心任务是判断这个新主题在语义上是否可以归入某个已存在的主题，或者它是否应该被视为一个全新的主主题。
    这个功能对于对话管理、知识库构建和上下文跟踪至关重要，它能帮助系统将相关的对话内容组织在一起。

    核心设计思想:
    -   **语义归一化 (Semantic Normalization)**: 核心目标是将语义相近但表述不同的主题进行归一化处理。
        例如，用户可能先问“如何连接数据库？”，后续又问“数据库的连接配置方法”，这两个主题在语义上是相同的。
        本智能体通过 LLM 的理解能力，将后者匹配到前者，避免了主题的碎片化。
    -   **动态与可配置 (Dynamic and Configurable)**: 通过 `prompt_manager` 从外部加载 Prompt 模板，
        使得匹配逻辑（即 Prompt 的内容）可以独立于代码进行修改和优化。这增强了系统的灵活性和可维护性，
        允许在不重新部署代码的情况下调整匹配策略。
    -   **鲁棒性与回退 (Robustness and Fallback)**: 当已存在的主题列表为空时，系统有一个明确的回退逻辑：
        直接将候选主题采纳为第一个主主题。这确保了系统在初始状态下也能正常工作。
    -   **结果清洗 (Result Cleansing)**: 从 LLM 返回的结果中移除了多余的引号和空白字符，
        确保输出的主题名称是干净、规范的，便于后续的程序处理。

    业务逻辑:
    1.  **处理空列表**: 检查 `existing_topics` 列表是否为空。如果为空，直接返回候选主题作为第一个主主题，无需调用 LLM。
    2.  **构建 Prompt**: 如果列表不为空，调用 `_build_prompt` 方法。该方法从 `prompt_manager` 获取模板，
        并将当前的候选主题和已存在的主题列表填充进去，生成一个结构化的 Prompt。
    3.  **调用 LLM**: 使用 `execute_llm_call` 函数，将构建好的 Prompt 和 LLM 配置（如模型、温度）发送给 LLM。
    4.  **清洗结果**: 对 LLM 返回的文本进行清洗，去除可能存在的引号和多余空格。
    5.  **返回决策**: 返回 LLM 决策出的主主题名称和调用统计数据。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化主题匹配智能体。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager
            
    def _build_prompt(self, candidate_topic: str, existing_topics: List[str]) -> List:
        """
        构建用于主题匹配的Prompt，现在从PromptManager加载模板。

        Args:
            candidate_topic (str): 需要进行匹配的候选主题字符串。
            existing_topics (List[str]): 已存在的主题列表。

        Returns:
            List: 构建好的、符合 LLM API 要求的消息列表。
        """
        # 将主题列表格式化为易于阅读的字符串
        topic_list_str = "\n".join([f"- {topic}" for topic in existing_topics])

        # 从 prompt_manager 加载并格式化系统提示
        system_prompt = self.prompt_manager.format_prompt(
            agent_name="topic_matching_agent",
            topic_list_str=topic_list_str,
            candidate_topic=candidate_topic
        )
        
        # 构建最终的消息结构
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": candidate_topic} # 用户内容通常是简单的重复，以触发LLM的响应
        ]
        return messages

    async def match_topic(
        self, 
        candidate_topic: str, 
        existing_topics: List[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        执行主题匹配的核心方法。

        此方法编排了整个匹配流程：处理边界情况（空列表），构建提示，调用LLM，并处理返回结果。

        Args:
            candidate_topic (str): 需要匹配的候选主题。
            existing_topics (List[str]): 已存在的主题列表，用于与候选主题进行比较。

        Returns:
            Tuple[str, Dict[str, Any]]:
                - `master_topic` (str): LLM 决策出的主主题。可能是候选主题本身，也可能是列表中的一个现有主题。
                - `stats` (Dict[str, Any]): LLM 调用的统计数据。
        """
        # 边界条件：如果还没有任何主主题，直接将候选主题采纳为第一个
        if not existing_topics:
            logger.info("已知主题列表为空，直接采用候选主题作为第一个主主题。")
            empty_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "N/A"}
            return candidate_topic, empty_stats

        # 构建 Prompt
        prompt = self._build_prompt(candidate_topic, existing_topics)
        # logger.info(f"调用主题匹配智能体，候选主题: '{candidate_topic}'，匹配列表: {existing_topics}")
        
        # 配置并执行 LLM 调用
        config = LLMConfig(model="gpt-4o", temperature=0.6)
        master_topic, stats = await execute_llm_call(prompt, config)
        
        # 清洗 LLM 的输出，去除可能包含的引号或多余空格
        master_topic = master_topic.strip().replace('"', '').replace("'", "")
        
        logger.info(f"TopicMatcherAgent: 决策的主题是: '{master_topic}'")
        return master_topic, stats