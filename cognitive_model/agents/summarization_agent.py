# cognitive-model/agents/summarization_agent.py

import logging
from typing import Dict, Any, List, Tuple
from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig, format_messages_for_llm
from cognitive_model.config.prompt_manager import PromptManager


logger = logging.getLogger("cognitive_model.summarization")

"""
总结智能体 (SummarizationAgent)

该智能体是认知模型实现长期记忆和上下文管理的关键组件。它的核心职责是在每次
对话交互后，以滚动更新的方式维护一个简洁、准确的对话摘要。这对于保持长期、
连贯的对话至关重要。

核心设计思想:
1.  **滚动摘要 (Rolling Summary)**: 为了在无限长的对话中保持上下文，同时避免
    Token 数量无限制增长，本智能体不从头生成摘要，而是在上一轮的摘要基础上，
    融入最新的对话内容。这种“滚动”机制极大地提高了效率和可扩展性。
2.  **上下文压缩**: 本质上，这是一个有损但高效的上下文压缩过程。它将冗长的、
    逐字逐句的对话历史，提炼成包含核心信息、关键决策和用户意图的精简文本。
    这使得在后续交互中，模型能以极低的Token成本快速“回忆”起之前的对话内容。
3.  **Prompt驱动**: 摘要的质量和风格完全由一个精心设计的系统提示（System Prompt）
    来控制。通过 `prompt_manager`，可以灵活地调整摘要的生成策略，例如，
    可以要求摘要更侧重于事实、情感、还是待办事项，而无需修改代码。
4.  **关注点分离**: SummarizationAgent 只负责“生成摘要”这一单一任务。它不关心
    对话的其它方面，如意图识别或工具调用。这种设计使得它成为一个可复用、
    易于维护的独立组件。
"""

class SummarizationAgent:
    """
    总结智能体 (SummarizationAgent)

    该智能体负责在每次对话交互后，以滚动方式更新对话的摘要。
    这对于维持长期对话的上下文至关重要，因为它能将冗长的对话历史压缩成一段精炼的文本，
    从而在后续的交互中减少 Token 消耗，并帮助模型快速把握对话的核心内容。

    业务逻辑:
    1.  **接收输入**: 接收上一轮的旧摘要 (`old_summary`) 和到目前为止的完整对话历史 (`full_history`)。
    2.  **构建提示**: 将对话历史格式化为纯文本，并使用 `prompt_manager` 来构建一个专门用于生成摘要的系统提示 (System Prompt)。这个提示会指导 LLM 如何在旧摘要的基础上，结合新的对话内容，生成一份更新后的摘要。
    3.  **准备消息**: 构建发送给 LLM 的消息列表，其中包含系统提示和作为用户输入的旧摘要。
    4.  **调用 LLM**: 使用 `execute_llm_call` 调用 LLM（如 `gpt-4o`），并获取生成的新摘要。
    5.  **返回结果**: 返回新生成的摘要文本以及调用过程中的统计数据。

    方法:
    - `summarize(old_summary, full_history)`: 异步方法，执行上述业务逻辑，生成并返回一个新的对话摘要和统计信息。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化总结智能体。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager

    
    async def summarize(
        self,
        old_summary: str,
        full_history: List[Dict[str, str]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        生成一个新的、更新后的对话摘要。

        业务逻辑:
        1.  **格式化历史**: 将结构化的对话历史（一个字典列表）转换成一个单一的、
            易于LLM阅读的纯文本字符串。每条消息都以“角色: 内容”的格式呈现。
        2.  **构建系统提示**: 调用 `prompt_manager`，将旧摘要和格式化后的对话历史
            注入到预定义的“summarization_agent”提示模板中。这个模板是整个摘要
            质量的核心，它会精确地指示LLM如何执行“滚动更新”任务。
        3.  **构建消息负载**: 创建一个符合OpenAI API规范的消息列表。值得注意的是，
            我们将旧摘要放在 `user` 角色的内容中。这是一种巧妙的技巧，引导模型
            将旧摘要视为需要处理和扩展的主要输入，而不是仅仅作为背景信息。
        4.  **配置与执行**: 创建一个 `LLMConfig` 实例，设置合适的模型（如 `gpt-4o`）
            和温度参数（`temperature=0.6`，以在创造性和事实性之间取得平衡）。
            然后，调用 `execute_llm_call` 来异步执行LLM调用。
        5.  **返回结果**: 返回LLM生成的新摘要文本，以及包含Token使用详情的统计字典。

        Args:
            old_summary (str): 上一轮对话结束时生成的摘要。如果是第一次生成，则为空字符串。
            full_history (List[Dict[str, str]]): 截止到目前为止的完整对话历史列表，
                                                 每个元素是一个包含 'role' 和 'content' 的字典。

        Returns:
            Tuple[str, Dict[str, Any]]:
                - `new_summary` (str): 新生成的、更新后的摘要文本。
                - `stats` (Dict[str, Any]): LLM 调用的统计数据，例如 Token 使用量。
        """
        logger.info("调用总结智能体以更新摘要...")
        # 将多轮对话历史拼接成一个字符串
        history_text = format_messages_for_llm(full_history)

        # 使用 prompt_manager 格式化生成摘要的系统提示
        # 这允许动态地将旧摘要和对话历史注入到预设的模板中
        system_prompt = self.prompt_manager.format_prompt(
            agent_name="summarization_agent",
            old_summary=old_summary,
            history_text=history_text
        )
        
        messages = format_messages_for_llm(system_prompt=system_prompt, history=history_text)
        print(f"总结消息: {messages}")
        # 配置并执行 LLM 调用
        config = LLMConfig(model="gpt-4o", temperature=0.6)
        new_summary, stats = await execute_llm_call(messages, config)
        
        logger.info("总结智能体已生成新摘要。")
        return new_summary, stats