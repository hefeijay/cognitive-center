# cognitive-model/agents/routing_agent.py

import logging
import json
import re
from typing import Dict, Any, Tuple, List

from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig, format_messages_for_llm, format_config_for_llm
from .formatting_agent import FormattingAgent
from cognitive_model.tools.tool_registry import ToolRegistry
from cognitive_model.config.prompt_manager import PromptManager

logger = logging.getLogger("app.cognitive_model.routing")

"""
路由智能体 (RoutingAgent)

这是认知模型中的核心决策智能体，扮演着“总调度师”或“交通警察”的角色。
它的核心职责是分析用户的输入，并决定处理该输入的最佳路径。这个决策是整个系统
能够处理复杂、多步骤任务，并有效利用外部工具的关键。

核心设计思想:
1.  **决策与执行分离**: RoutingAgent 只负责“决策”，即判断应该使用哪个工具
    或直接回答。它不亲自执行工具调用或生成最终答案，而是将决策结果传递给
    Orchestrator，由后者负责执行。这遵循了单一职责原则，使得系统结构更清晰。
2.  **动态工具集成**: 通过与 `tool_registry` 模块的交互，RoutingAgent 能够
    动态地获取所有当前可用的工具及其描述。这意味着系统可以轻松地通过注册
    新工具来扩展其功能，而无需修改 RoutingAgent 的核心逻辑。
3.  **两阶段LLM调用**: 为了确保从LLM获得稳定、可靠的结构化输出（JSON格式），
    采用了“初步决策 + 格式化清洗”的两阶段流程。第一步让LLM自由发挥，生成
    初步想法；第二步利用 `FormattingAgent` 将其强制转换为严格的JSON格式。
    这大大提高了系统的鲁棒性。
4.  **鲁棒性与回退**: 设计了多层保护机制来处理潜在的失败情况。
    -   使用正则表达式提取JSON，以应对LLM输出中夹杂的额外文本。
    -   对解析后的JSON进行严格的结构验证。
    -   在格式化或解析失败时，能够优雅地回退到“直接回答”的安全模式，
        确保系统总能给出一个合理的响应，而不是彻底崩溃。
"""

class RoutingAgent:
    """
    路由智能体 (v2.0)

    负责分析用户输入，并决定最佳的处理路径。它可以决策是直接回答，
    还是调用一个或多个工具来完成用户的请求。这是实现复杂任务处理和
    工具集成的关键决策点。
    """
    def __init__(self, prompt_manager: PromptManager, tool_registry: ToolRegistry):
        """
        初始化路由智能体。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager
        self.tool_registry = tool_registry
        self.formatting_agent = FormattingAgent(prompt_manager)

    # def _build_messages(self, user_input: str, history: List[Dict[str, str]], session_tools: Dict[str, Any]) -> List:
    #     """
    #     构建用于路由决策的系统提示(System Prompt)。

    #     业务逻辑:
    #     1.  **获取工具描述**: 调用 `tool_registry.get_all_tool_descriptions()`
    #         方法，动态获取所有已注册工具的最新、最完整的描述列表。
    #     2.  **加载提示模板**: 从 `prompt_manager` 获取为 `routing_agent` 设计的
    #         专用提示模板。
    #     3.  **格式化填充**: 将动态获取的工具描述和当前的用户输入填充到模板中，
    #         生成一个内容完整、上下文丰富的系统提示。
    #     4.  **构建消息列表**: 将生成的系统提示和用户输入组装成符合OpenAI API
    #         要求的消息列表格式。

    #     Args:
    #         user_input (str): 用户的原始输入文本。
    #         session_tools (Dict[str, Any]): 本次会话可用的工具集。

    #     Returns:
    #         List: 构建好的消息列表，可直接用于LLM调用。
    #     """
    #     tool_descriptions = self.tool_registry.get_all_tool_descriptions(tools=session_tools)
    #     system_prompt = self.prompt_manager.format_prompt(
    #         agent_name="routing_agent",
    #         tool_descriptions=tool_descriptions,
    #         user_input=user_input
    #     )
    #     if system_prompt:
    #         logger.info("RoutingAgent: 准备构建决策提示...")
    #         # 格式化历史记录
    #         messages = format_messages_for_llm(history)
    #         # formatted_history = []
    #         # if history:
    #         #     formatted_history = [{"role": msg.role, "content": msg.content} for msg in history]

    #         # messages = [{"role": "system", "content": system_prompt}]
    #         # messages.extend(formatted_history)
    #     return messages

    def _extract_json_from_response(self, raw_response: str) -> str:
        """
        从LLM的原始响应中稳定地提取出JSON代码块。

        业务逻辑:
        -   使用一个强大的正则表达式来匹配两种常见的JSON返回模式：
            1.  被Markdown代码块（```json ... ```）包裹的JSON。
            2.  直接返回的裸JSON对象（{...}）。
        -   `re.DOTALL` 标志确保了正则表达式可以处理跨越多行的JSON字符串。
        -   如果匹配成功，则返回第一个或第二个捕获组的内容；如果匹配失败，
            则返回原始响应，以供后续步骤处理。

        Args:
            raw_response (str): LLM返回的原始字符串。

        Returns:
            str: 提取出的纯JSON字符串，或在无法提取时返回原始字符串。
        """
        match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', raw_response, re.DOTALL)
        return match.group(1) or match.group(2) if match else raw_response

    def _validate_decision(self, decision: Dict[str, Any]) -> None:
        """
        对解析后的JSON决策进行结构和内容的验证。

        业务逻辑:
        1.  **检查核心字段**: 验证决策字典中是否包含必需的 `route` 字段。
        2.  **验证路由类型**: 确保 `route` 字段的值是预定义集合 `["direct_answer", "tool_use"]`
            中的一个。
        3.  **检查工具使用字段**: 如果 `route` 的值是 `tool_use`，则进一步验证
            是否包含 `tool` 和 `mode` 字段。
        4.  **抛出异常**: 如果任何验证失败，就抛出 `ValueError` 并附带清晰的
            错误信息。这个异常将在上层调用栈中被捕获，并触发回退机制。

        Args:
            decision (Dict[str, Any]): 解析后的决策字典。

        Raises:
            ValueError: 如果决策的结构或内容不符合预定义规范。
        """
        if "route" not in decision or decision["route"] not in ["direct_answer", "tool_use"]:
            raise ValueError("字段 'route' 缺失或无效。")
        if decision["route"] == "tool_use":
            if "tool" not in decision or "mode" not in decision:
                raise ValueError("字段 'tool' 或 'mode' 缺失。")

    async def run(self, user_input: str, history: List[Dict[str, str]], session_tools: Dict[str, Any], model_config: dict, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
            路由决策的核心执行函数，编排整个决策流程。

            业务逻辑:
            1.  **构建提示**: 调用 `_build_prompt` 方法，为当前用户输入构建一个
                包含最新工具信息的完整提示。
            2.  **初步决策 (LLM Call #1)**: 使用一个较为自由的配置（temperature=0.6）
                调用LLM，获取一个初步的、可能不完全符合格式的路由建议。
            3.  **格式化清洗 (LLM Call #2)**: 将第一步的原始输出和目标JSON格式描述
                一起传递给 `FormattingAgent`。`FormattingAgent` 会进行第二次LLM调用，
                其唯一目标就是将输入强制转换为干净、标准的JSON字符串。
            4.  **统计聚合**: 累加两次LLM调用的Token消耗，以便进行成本追踪。
            5.  **解析与验证**: 
                -   尝试使用 `json.loads` 解析格式化后的字符串。
                -   如果解析成功，则调用 `_validate_decision` 进行结构验证。
                -   如果一切顺利，记录成功的决策并返回结果。
            6.  **异常处理与回退**: 如果在解析或验证过程中发生任何错误（`JSONDecodeError`
                或 `ValueError`），则捕获异常，记录详细的错误日志，并生成一个安全
                的回退决策（`direct_answer`），确保系统流程不会中断。

            Args:
                user_input (str): 用户的原始输入文本。
                session_tools (Dict[str, Any]): 本次会话可用的工具集。

            Returns:
                Tuple[Dict[str, Any], Dict[str, Any]]:
                    - 第一个元素是最终的、经过验证的决策字典。
                    - 第二个元素是本次操作的Token消耗统计。
        """
        # 第一次调用：生成初步决策
        # logger.info("RoutingAgent: 调用 LLM 获取初步路由建议...")
        
        tool_descriptions = self.tool_registry.get_all_tool_descriptions(tools=session_tools)
        system_prompt = self.prompt_manager.format_prompt(
            agent_name="routing_agent",
            tool_descriptions=tool_descriptions,
            user_input=user_input
        )
        messages = format_messages_for_llm(system_prompt, history)
        config = format_config_for_llm(model_config)
        
        raw_decision, stats = await execute_llm_call(messages, config)

        # 第二次调用：格式化输出
        logger.info("RoutingAgent: 调用 FormattingAgent 清洗输出...")
        target_format = """
        你的输出必须是一个JSON对象，不含任何其他文本。
        结构为: {"route": "...", "tool": "...", "mode": "...", "args": {...}, "reason": "..."}。
        route 的值必须是 "direct_answer" 或 "tool_use"。
        """
        formatted_str, format_stats = await self.formatting_agent.run(raw_decision, target_format)

        # 累加 Token 消耗
        stats["total_tokens"] = stats.get("total_tokens", 0) + format_stats.get("total_tokens", 0)
        stats["formatting_agent_tokens"] = format_stats.get("total_tokens", 0)

        # 最终解析与验证
        try:
            decision = json.loads(formatted_str)
            self._validate_decision(decision)
            logger.info(f"RoutingAgent: 决策: {decision.get('route')}")
            print(f"完整决策：{decision}")

            return decision, stats
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"RoutingAgent: 决策失败 - {e}. 原始输出: {formatted_str}")
            fallback = {"route": "direct_answer", "reason": "格式化失败，已回退为直接回答。"}
            return fallback, stats
