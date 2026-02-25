# cognitive_model/agents/intent_agent.py

import logging
from typing import Dict, Any, Tuple, List, Optional

from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig, format_messages_for_llm, format_config_for_llm
from cognitive_model.config.prompt_manager import PromptManager

logger = logging.getLogger("app.cognitive_model.intent")

class IntentAgent:
    """
    意图识别智能体 (IntentAgent)

    该智能体在认知模型中扮演着"前门"或"分诊台"的角色。它的核心职责是分析用户的
    原始输入，并快速判断其核心意图。这是整个系统进行决策和任务分发的第一步，
    其准确性对后续所有处理流程的效率和正确性至关重要。

    核心设计思想:
    1.  **分类与路由**: 它是策略模式（Strategy Pattern）的入口。通过将用户的模糊输入
        映射到一个预定义的、标准化的意图（如"提问"、"反馈"、"调整"等），为
        `CognitiveOrchestrator`（认知协调器）选择正确的处理器（Handler）提供了依据。
    2.  **简化复杂性**: 它将"理解用户想做什么"这一复杂的自然语言理解任务，简化并封装成
        一个单一职责的组件。这使得系统的其他部分不必关心语言的细微差别，只需根据
        明确的意图标签来行动。
    3.  **鲁棒性与回退**: 意图识别不可能100%准确。因此，设计中包含了验证和回退机制。
        如果 LLM 返回的意图不在预定义的列表中，系统会记录一个警告，并将其归类为最
        常见或最安全的默认意图（例如"提问"），确保系统总能以一种可预测的方式继续运行，
        避免因无法识别意图而完全卡住。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化意图识别智能体。

        Args:
            prompt_manager (PromptManager): 一个 PromptManager 的实例，用于获取和格式化提示。
        """
        self.prompt_manager = prompt_manager
            
    async def get_intent(self, user_input: str, history: Optional[List[Dict[str, str]]] = None, model_config: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        调用 LLM 来确定用户输入的核心意图，并结合对话历史上下文。

        这是该 Agent 的核心公共方法。它负责执行完整的意图识别流程。

        业务逻辑:
        1.  **构建系统提示**: 从 `prompt_manager` 获取专门为意图识别设计的 System Prompt。
            这个 Prompt 通常会包含所有有效的意图列表和分类准则，以指导 LLM 进行选择。
        2.  **整合对话历史**: 将传入的 `history` (如果存在) 构建到发送给 LLM 的消息列表中，
            为模型提供必要的上下文。
        3.  **配置与调用LLM**: 根据传入的 `model_config` 或默认值设置 LLM 的配置, 
            并使用 `execute_llm_call` 发起请求。
        4.  **结果清洗**: 对 LLM 返回的原始文本进行 `.strip()` 和去除引号等操作，以获得
            一个干净的意图字符串。这是必要的，因为 LLM 的输出可能包含多余的格式。
        5.  **有效性验证**: 检查清洗后的意图字符串是否存在于预定义的 `valid_intents` 列表中。
        6.  **回退处理**: 如果验证失败，将意图强制设置为默认值（"提问"），并记录警告。
            这增强了系统的容错能力。
        7.  **返回结果**: 返回最终确定的意图字符串和包含 Token 消耗的统计信息。

        Args:
            user_input (str): 用户输入的原始文本。
            history (Optional[List[Dict[str, str]]], optional): 对话历史列表，
                其中每个元素都是一个包含 'role' 和 'content' 的字典。默认为 None。
            model_config (Optional[Dict[str, Any]], optional): 用于覆盖默认值的LLM配置。
                例如: `{'model_name': 'gpt-4o-mini', 'temperature': 0.7}`.

        Returns:
            Tuple[str, Dict[str, Any]]: 一个元组，包含识别出的意图字符串和 LLM 调用的统计信息。
        """
        system_prompt = self.prompt_manager.format_prompt(
            agent_name="intent_agent",
            user_input=user_input
        )
        
        if system_prompt:
            # logger.info(f"调用意图智能体以确定输入类型 (包含历史记录)...")
            # 使用新的公共方法格式化配置
            config = format_config_for_llm(model_config)
            # 使用新的公共方法格式化消息
            messages = format_messages_for_llm(system_prompt, history or [])
            
            intent, stats = await execute_llm_call(messages, config)
            
            cleaned_intent = intent.strip().replace('"', '').replace("'", "")
            
            valid_intents = ["提问", "反馈", "通知", "调整", "唤醒", "自我介绍"]
            if cleaned_intent not in valid_intents:
                logger.warning(f"意图智能体返回意外结果: '{cleaned_intent}'，将默认其为'提问'。")
                cleaned_intent = "提问"  
            
            logger.info(f"IntentAgent: 用户意图为: '{cleaned_intent}'")
            return cleaned_intent, stats
            
        return "提问", {}