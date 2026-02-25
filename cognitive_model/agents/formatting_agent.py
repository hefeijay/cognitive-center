import logging
from typing import Dict, Any, Tuple, List

from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig
from cognitive_model.config.prompt_manager import PromptManager
logger = logging.getLogger("app.cognitive_model.formatting")


class FormattingAgent:
    """
    格式化智能体 (FormattingAgent)

    这是一个专门用于对大语言模型（LLM）的输出进行结构化清洗和格式化的工具型智能体。
    在复杂的系统中，LLM 的输出有时可能不稳定，会包含额外的解释性文本、代码块标记或其他
    非预期的内容。`FormattingAgent` 的核心职责就是将这些"原始"、"不干净"的输出，
    强制转换为一个严格符合预定义格式（通常是 JSON）的干净字符串。

    核心设计思想:
    1.  **二次调用 (Two-pass approach)**: 它是对 LLM 的一次"元调用"。第一次调用由其他 Agent
        完成，获取业务结果的草案；当需要严格格式时，将该草案作为输入，进行第二次调用，
        专门负责格式化。这是一种用 LLM 的能力来约束其自身输出的有效策略。
    2.  **可靠性保障**: 在需要程序化处理 LLM 输出的场景（例如，解析 JSON 以进行后续操作）中，
        此 Agent 是保障系统稳定性的关键。它确保了无论上游 LLM 的输出如何波动，下游模块
        接收到的总是一个格式正确、可预测的输入。
    3.  **关注点分离**: 将格式化的逻辑从其他业务型 Agent（如 `CognitiveTunerAgent`）中
        剥离出来，使得业务 Agent 可以更专注于其核心任务，而将输出的最终呈现和规范化
        交给 `FormattingAgent`。这提高了代码的模块化和可重用性。
    """

    def __init__(self, prompt_manager: PromptManager):
        """
        初始化格式化代理。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager

    async def run(self, raw_input: str, target_format_description: str) -> Tuple[str, Dict[str, Any]]:
        """
        接收一个不规范的 LLM 输出，并根据目标格式描述进行标准化。

        这是该 Agent 的核心公共方法。它封装了调用 LLM 进行格式化的完整流程。

        业务逻辑:
        1.  **构建消息**: 调用 `_build_messages` 方法，根据原始输入和目标格式描述，
            创建一个专门用于格式化任务的 System Prompt 和 User Message。
        2.  **调用LLM**: 使用 `execute_llm_call` 向 LLM 发起请求。这里的 LLM 调用
            目的不是创造新内容，而是对已有内容进行"重塑"。
        3.  **清洗与返回**: 对 LLM 返回的格式化结果进行 `.strip()` 清理，去除可能存在
            的前后空白符，然后返回干净的字符串和 Token 统计信息。
        4.  **异常处理**: 如果 LLM 调用失败，会捕获异常，记录错误，并返回一个空的
            JSON 对象字符串 `"{}"`，以确保下游调用方总能得到一个可解析的字符串，
            避免因 `None` 或异常导致整个流程崩溃。

        Args:
            raw_input (str): 初始 LLM 的原始回复（可能是非结构化的）。
            target_format_description (str): 一个清晰的描述，告诉 LLM 应该生成什么格式，
                                           例如一个 JSON 结构的模板。

        Returns:
            Tuple[str, Dict[str, Any]]: 一个元组，包含:
                - 格式化后的字符串。
                - LLM 调用的统计数据（或在出错时包含错误信息的字典）。
        """
        messages = self._build_messages(raw_input, target_format_description)

        # logger.info("调用格式化智能体进行输出标准化...")
        config = LLMConfig(model="gpt-4o", temperature=0.6)

        try:
            formatted_output, stats = await execute_llm_call(messages, config)
            formatted_output = formatted_output.strip()

            logger.debug(f"格式化后的内容:\n{formatted_output}")
            return formatted_output, stats

        except Exception as e:
            logger.exception(f"格式化调用失败: {e}")
            return "{}", {"error": str(e)}

    def _build_messages(self, raw_input: str, target_format_description: str) -> List[Dict[str, str]]:
        """
        构建用于格式化任务的 messages 列表 (system + user)。

        业务逻辑:
        1.  **加载模板**: 使用 `prompt_manager` 加载专门为 `FormattingAgent` 设计的
            System Prompt 模板。
        2.  **填充模板**: 将 `raw_input` (原始输出) 和 `target_format_description` (目标格式)
            作为变量填充到模板中，生成最终的 System Prompt。
        3.  **构建列表**: 创建一个包含 System Prompt 和 User Message (即 `raw_input`)
            的列表，这是符合 OpenAI API 格式的标准输入。
        4.  **降级处理**: 如果 `prompt_manager` 加载失败，会使用一个通用的、硬编码的
            默认提示，以保证基础功能可用。

        Args:
            raw_input (str): LLM 的原始输出。
            target_format_description (str): 结构化格式的描述。

        Returns:
            List[Dict[str, str]]: 用于 LLM 调用的 messages 列表。
        """
        system_prompt = self.prompt_manager.format_prompt(
            agent_name="formatting_agent",
            raw_input=raw_input,
            target_format_description=target_format_description
        )

        if not system_prompt or system_prompt.startswith("无法"):
            logger.warning(f"格式化智能体获取 system prompt 失败: {system_prompt}")
            system_prompt = "请将以下文本转换为一个干净、结构化的 JSON 格式。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_input}
        ]

        logger.debug(f"构建的格式化 messages:\n{messages}")
        return messages
