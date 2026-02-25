# cognitive-model/agents/thinking_agent.py

import logging
from typing import Dict, Any, List, Tuple
from cognitive_model.agents.llm_utils import execute_llm_call, execute_llm_call_stream, LLMConfig, format_messages_for_llm
from cognitive_model.config.prompt_manager import PromptManager


logger = logging.getLogger("app.cognitive_model.thinking")

class ThinkingAgent:
    """
        思考智能体 (ThinkingAgent)

        该智能体负责执行一个已经由 Orchestrator 构建好的、完整的 Prompt。
        它的核心职责是调用底层的大语言模型（LLM）来生成最终的、面向用户的回复。
        这个过程是认知流程的最后一步，将所有处理过的信息和决策转化为自然语言输出。

        核心设计思想:
        -   **最终执行者 (Final Executor)**: `ThinkingAgent` 是整个认知链条的最后一步。它不负责决策、路由或信息处理，
            只负责将最终构建好的、完整的 Prompt 发送给 LLM，并获取最终的自然语言回复。这种设计确保了关注点分离，
            使得其他智能体可以专注于各自的任务，而 `ThinkingAgent` 则专注于与 LLM 的最终交互。
        -   **Prompt 完整性 (Prompt Integrity)**: `ThinkingAgent` 假设接收到的 `messages` 已经是完整且经过精心设计的。
            它不修改或构建 Prompt，而是直接使用。这依赖于上游的 `Orchestrator` 或其他调用者已经根据 AI 宪法、
            对话历史、工具调用结果等信息构建了一个高质量的 Prompt。
        -   **简单与直接 (Simplicity and Directness)**: 该智能体的实现非常直接，核心逻辑就是调用 `execute_llm_call`。
            这种简单性降低了出错的可能性，并使其易于维护和理解。所有复杂性都已经被上游的智能体处理完毕。
        -   **可配置的 LLM 调用 (Configurable LLM Call)**: 尽管 `ThinkingAgent` 本身很简单，但它通过 `LLMConfig`
            来配置底层的 LLM 调用，允许在运行时动态调整模型、温度等参数。这为未来的实验和优化
            （例如，针对不同任务使用不同模型或配置）提供了灵活性。

        业务逻辑:
        1.  **接收消息**: 从 Orchestrator 接收一个已经完全构建好的消息列表（`messages`），这个列表通常包含了系统提示、历史对话、上下文信息以及当前的提问。
        2.  **配置 LLM**: 设置调用 LLM 所需的配置，例如模型名称（如 "gpt-4o"）和温度（`temperature`），温度控制了生成文本的创造性。
        3.  **调用 LLM**: 使用 `execute_llm_call` 工具函数，将消息列表和配置传递给 LLM，并等待其返回结果。
        4.  **返回结果**: 返回 LLM 生成的文本回复以及调用过程中的统计数据（如 Token 使用量）。

        方法:
        - `think(messages)`: 异步方法，接收一个消息列表，调用 LLM 并返回其生成的回复和统计信息。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化思考智能体。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager
        
    async def run(self, user_input: str, history: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
            执行思考任务，调用 LLM 生成回复。

            这个方法是 `ThinkingAgent` 的核心，它接收由 Orchestrator 根据 AI 宪法、用户历史、上下文信息等
            精心构建的完整消息列表，然后调用 LLM 来生成最终的答复。

            Args:
                user_input (str): 用户的当前输入。
                history (List[Dict[str, str]]): 之前的对话历史。
                **kwargs: 包含其他可选参数的字典，例如 `tool_result`。

            Returns:
                Tuple[str, Dict[str, Any]]:
                    - `response` (str): LLM 生成的原始文本回复。
                    - `stats` (Dict[str, Any]): LLM 调用的统计数据，例如 `{"total_tokens": 1024, "prompt_tokens": 512, "completion_tokens": 512}`。
        """
        logger.info("调用思考智能体执行任务...")
        tool_result = kwargs.get("tool_result")
        
        common_args = {
            "user_input": user_input,
        }
                
        if tool_result:
            system_prompt = self.prompt_manager.format_prompt(
                agent_name="thinking_agent", template_key="with_tool", tool_result=tool_result, **common_args
            )
        else:
            system_prompt = self.prompt_manager.format_prompt(
                agent_name="thinking_agent", template_key="without_tool", **common_args
            )
        
        messages = format_messages_for_llm(system_prompt, history or [])
        
        config = LLMConfig(model="gpt-4o", temperature=0.6)
        
        response, stats = await execute_llm_call(messages, config)
        
        logger.info(f"思考智能体已生成回复: {response[:100]}...")
        return response, stats
    
    async def run_stream(self, user_input: str, history: List[Dict[str, str]], stream_callback=None, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
            执行流式思考任务，调用 LLM 生成流式回复。

            这是 ThinkingAgent 的流式版本，它在原有功能基础上增加了实时流式响应能力。
            通过回调函数机制，可以将 LLM 生成的内容片段实时推送给调用方，从而实现更好的用户体验。

            Args:
                user_input (str): 用户的当前输入。
                history (List[Dict[str, str]]): 之前的对话历史。
                stream_callback (callable, optional): 流式回调函数，接收参数：
                    - content (str): 当前接收到的内容片段
                    - is_final (bool): 是否为最终完成状态
                    - full_content (str): 到目前为止的完整内容
                **kwargs: 包含其他可选参数的字典，例如 `tool_result`。

            Returns:
                Tuple[str, Dict[str, Any]]:
                    - `response` (str): LLM 生成的完整文本回复。
                    - `stats` (Dict[str, Any]): LLM 调用的统计数据。
        """
        logger.info("调用思考智能体执行流式任务...")
        tool_result = kwargs.get("tool_result")
        
        common_args = {
            "user_input": user_input,
        }
                
        if tool_result:
            system_prompt = self.prompt_manager.format_prompt(
                agent_name="thinking_agent", template_key="with_tool", tool_result=tool_result, **common_args
            )
        else:
            system_prompt = self.prompt_manager.format_prompt(
                agent_name="thinking_agent", template_key="without_tool", **common_args
            )
        
        messages = format_messages_for_llm(system_prompt, history or [])
        
        config = LLMConfig(model="gpt-4o-mini", temperature=0.6)
        
        # 使用流式版本的 LLM 调用
        response, stats = await execute_llm_call_stream(messages, config, stream_callback)
        
        logger.info(f"思考智能体已完成流式回复: {response}")
        return response, stats