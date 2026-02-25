import logging
import json
from typing import Dict, Any, Tuple, Optional, List

from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig
from .formatting_agent import FormattingAgent
from cognitive_model.config.prompt_manager import PromptManager
from ..tools.native.file_editor import read_file

logger = logging.getLogger("app.cognitive_model.cognitive_tuner")


class CognitiveTunerAgent:
    """
    认知修正智能体 (CognitiveTunerAgent)

    该智能体是实现模型自我进化和优化的核心组件。它专门负责处理用户的反馈，
    特别是那些关于模型行为、响应质量或能力边界的反馈。通过分析这些反馈，
    它能够智能地生成针对其他智能体（Agent）的系统提示（System Prompt）的修改建议。

    核心设计思想:
    1.  **元认知 (Metacognition)**: 它是模型"思考如何思考"能力的体现。它不直接回答用户问题，
        而是通过修改其他 Agent 的指导原则（Prompt）来间接优化整个系统的未来行为。
    2.  **闭环学习**: 用户的反馈 -> `CognitiveTunerAgent` 分析 -> 生成修改计划 -> 用户确认 ->
        应用修改。这个流程形成了一个完整的闭环，使模型能够根据用户输入持续学习和改进。
    3.  **安全与可控**: 修改计划不是自动执行的，而是以结构化的 JSON 格式呈现给用户，
        必须经过用户的明确批准（例如，回复"同意"）后，才由 `TuningHandler` 应用。这确保了
        所有对模型核心行为的修改都在人类监督下进行，防止了意外或恶意的修改。
    4.  **关注点分离**: 将"生成修改计划"的逻辑封装在此 Agent 中，而将"执行修改"的逻辑
        放在 `TuningHandler` 中，遵循了单一职责原则，使得代码结构更清晰。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化认知修正智能体。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager
        self.formatting_agent = FormattingAgent(prompt_manager)

    async def generate_update_plan(
        self,
        user_input: str,
        model: str = "gpt-4o",
        temperature: float = 0.6
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        基于用户反馈，生成一个结构化的 Prompt 修改计划。

        这是该 Agent 的核心公共方法。它编排了一系列步骤，将非结构化的用户反馈
        转化为一个精确、可执行的 JSON 修改计划。

        业务逻辑:
        1.  **加载现有配置**: 调用 `_load_prompt_config` 从 `prompts.json` 文件中读取
            当前所有 Agent 的 Prompt 配置。这是 LLM 理解当前系统状态所必需的上下文。
        2.  **构建系统提示**: 调用 `_build_system_prompt`，将用户反馈和所有现有 Prompt
            整合进一个专门为本 Agent 设计的 System Prompt 中。这个 Prompt 会指示 LLM
            扮演一个"AI系统配置专家"的角色，分析反馈并提出具体的修改建议。
        3.  **调用LLM生成草案**: 使用 `execute_llm_call` 向 LLM 发起请求，生成一个
            初步的、可能是非结构化的修改计划草案。
        4.  **格式化清洗**: 将 LLM 的原始输出传递给 `_clean_plan_with_formatting_agent`。
            该方法会利用 `FormattingAgent` 强制将草案内容转换为严格的 JSON 格式，
            确保后续处理的可靠性。
        5.  **解析最终计划**: 调用 `_parse_clean_plan` 将格式化后的 JSON 字符串解析为
            Python 字典。
        6.  **返回结果**: 返回解析后的修改计划字典和包含 Token 消耗的统计信息。

        Args:
            user_input (str): 用户的原始反馈文本。
            model (str): 用于生成计划的 LLM 模型名称。
            temperature (float): LLM 的温度参数，用于控制生成结果的创造性。

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: 一个元组，包含:
                - 修改计划的字典，如果失败则包含 'error' 键。
                - LLM 调用的统计数据。
        """
        logger.info(f"认知修正智能体启动，分析用户反馈: '{user_input}'")

        all_prompts_json_string = self._load_prompt_config()
        if not all_prompts_json_string:
            return {"error": "无法读取内部配置，无法生成修正计划。"}, {}

        system_prompt = self._build_system_prompt(user_input, all_prompts_json_string)
        if not system_prompt:
            return {"error": "构建系统提示失败"}, {}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        logger.info("调用 LLM 生成初步修改计划...")
        config = LLMConfig(model=model, temperature=temperature)
        raw_plan, stats = await execute_llm_call(messages, config)

        logger.debug(f"原始修改计划内容:\n{raw_plan}")

        clean_json_str, format_stats = await self._clean_plan_with_formatting_agent(raw_plan)
        stats["total_tokens"] = stats.get("total_tokens", 0) + format_stats.get("total_tokens", 0)
        stats["formatting_agent_tokens"] = format_stats.get("total_tokens", 0)

        update_plan = self._parse_clean_plan(clean_json_str)
        if "error" in update_plan:
            return update_plan, stats

        logger.info(f"成功生成并解析了修正计划: {update_plan}")
        return update_plan, stats

    def _load_prompt_config(self) -> Optional[str]:
        """辅助方法：加载 prompts.json 配置文件。"""
        content = read_file("prompts_config")
        if content.startswith("错误："):
            logger.error("无法读取 prompts 配置文件。")
            return None
        return content

    def _build_system_prompt(self, user_input: str, all_prompts_json_string: str) -> Optional[str]:
        """辅助方法：构建用于生成修改计划的系统提示。"""
        prompt = self.prompt_manager.format_prompt(
            agent_name="cognitive_tuner_agent",
            user_feedback=user_input,
            all_prompts_json_string=all_prompts_json_string
        )
        if not prompt or prompt.startswith("无法") or prompt.startswith("格式化提示"):
            logger.error(f"系统提示构建失败: {prompt}")
            return None
        return prompt

    async def _clean_plan_with_formatting_agent(self, raw_plan: str) -> Tuple[str, Dict[str, Any]]:
        """辅助方法：使用 FormattingAgent 清洗和格式化 LLM 的原始输出。"""
        target_format_desc = """
        你的输出必须是一个JSON对象，不含任何其他文本。
        其结构必须是: {
            "analysis": "...",
            "target_file_key": "prompts_config",
            "target_agent": "...",
            "target_prompt_key": "..." or null,
            "proposed_new_prompt": "..."
        }
        """
        logger.info("调用 FormattingAgent 对计划进行清洗...")
        return await self.formatting_agent.run(raw_plan, target_format_desc)

    def _parse_clean_plan(self, clean_str: str) -> Dict[str, Any]:
        """辅助方法：将干净的 JSON 字符串解析为 Python 字典。"""
        try:
            return json.loads(clean_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"解析格式化计划失败: {e}. 内容:\n{clean_str}")
            return {"error": "生成修正计划失败，无法解析最终输出。"}
