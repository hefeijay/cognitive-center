# cognitive-model/agents/llm_utils.py

import os
import logging
from typing import Dict, Any, Tuple, List, Literal, Protocol,Optional
from dotenv import load_dotenv

from openai import AsyncOpenAI

import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()

logger = logging.getLogger("app.cognitive_model.llm_utils")


"""
LLM 调用工具库 (llm_utils)

这个模块是认知模型与大语言模型（LLM）交互的核心基础设施。它提供了一套完整的工具和抽象，
使得系统中的其他组件（如各种智能体）能够以一种统一、安全和可靠的方式与 LLM 进行通信。

核心设计思想:
1.  **统一抽象**: 通过 `LLMConfig` 类和 `execute_llm_call` 函数，为所有 LLM 调用提供
    了一个统一的接口。这使得切换底层模型或调整调用参数变得简单，同时确保了所有调用
    都遵循相同的最佳实践。
2.  **可靠性保障**: 集成了自动重试机制（使用 tenacity），以优雅地处理网络问题和临时
    错误。采用指数退避策略，在保持系统响应性的同时，避免对 API 服务造成过大压力。
3.  **成本控制**: 通过 tiktoken 库精确计算 Token 使用量，并在每次调用中返回详细的
    统计信息。这使得系统能够实时监控 API 使用成本，并在必要时采取措施（如截断过长输入）。
4.  **优雅降级**: 在配置加载和 API 调用等关键点都实现了合理的降级策略。例如，在
    tokenizer 初始化失败时回退到通用编码器，在 API 调用失败时返回友好的错误消息。
5.  **可观测性**: 通过详细的日志记录，使系统运行状态和潜在问题变得可见和可追踪。
    这对于问题诊断和性能优化至关重要。
"""

# --- 初始化 OpenAI API ---
# 从环境变量中获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # 如果未设置密钥，则抛出异常，因为这是与 LLM 通信的必要条件
    raise ValueError("OPENAI_API_KEY 环境变量未设置！")

aclient = AsyncOpenAI(api_key=api_key)

# --- 初始化 Tokenizer ---
try:
    # 优先尝试加载特定模型（如 gpt-4o-mini）的编码器，以获得最精确的 Token 计算
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    # 如果失败（例如，模型名称不存在或库版本问题），则回退到通用的 cl100k_base 编码器
    # cl100k_base 兼容 GPT-3.5 和 GPT-4 系列的大多数模型
    encoding = tiktoken.get_encoding("cl100k_base")


class LLMConfig:
    """
    LLM 调用配置类 (LLMConfig)

    这个类封装了调用大语言模型时的关键配置参数。它的设计目标是使 LLM 调用的配置
    过程更加结构化和类型安全，同时保持足够的灵活性以适应不同场景的需求。

    核心设计思想:
    1.  **参数封装**: 将分散的配置参数组织成一个内聚的对象，使得配置的传递和修改
        更加清晰和可控。
    2.  **默认值优化**: 为常用参数提供经过实践验证的默认值，在保持灵活性的同时，
        降低了使用门槛。
    3.  **可扩展性**: 虽然目前主要关注模型名称和温度参数，但类的结构允许在需要时
        轻松添加新的配置项（如 max_tokens、top_p 等）。

    Attributes:
        model (str): 要使用的 LLM 模型名称，默认为 "gpt-4o-mini"。
        temperature (float): 控制生成文本的随机性。值越高，输出越具创造性；值越低，
                           输出越确定。默认为 0.7。
    """
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature


class HasRoleAndContent(Protocol):
    """
    定义一个协议，用于表示具有 'role' 和 'content' 属性的对象。
    这允许 `format_messages_for_llm` 函数接受任何符合此结构的对象列表，
    """
    role: str
    content: str

def format_config_for_llm(
    model_config: Optional[Dict[str, Any]] = None
) -> LLMConfig:
    """
    """
    if model_config and model_config.get('model_name'):
        config = LLMConfig(
            model=model_config.get('model_name', 'gpt-4o'),
            temperature=model_config.get('temperature', 0.6)
        )
    else:
        config = LLMConfig(model="gpt-4o", temperature=0.6)
    return config

def format_messages_for_llm(
    system_prompt: str, 
    history: List[Any]
) -> List[Dict[str, str]]:
    """
    格式化聊天历史记录和系统提示，以符合LLM API的输入要求。

    此函数将系统提示和一系列聊天历史记录对象转换为一个字典列表，
    每个字典代表一条消息，包含 'role' 和 'content' 键。
    这是调用 `execute_llm_call` 之前准备输入数据的标准方法。

    Args:
        system_prompt (str): 要在对话开始时发送给LLM的系统级指令。
        history (List[Any]): 一个包含聊天历史记录的对象列表或字典列表。
                                           每个元素必须具有 'role' 和 'content' 属性或键。

    Returns:
        List[Dict[str, str]]: 一个格式化的消息列表，可直接用于LLM API调用。
    """
    # 格式化历史记录以符合LLM API要求
    formatted_history = []
    if history:
        # 兼容处理字典列表和对象列表
        if isinstance(history[0], dict):
            formatted_history = [{"role": msg.get("role"), "content": msg.get("content")} for msg in history]
        else:
            formatted_history = [{"role": msg.role, "content": msg.content} for msg in history]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(formatted_history)
    return messages


def count_tokens(text: str) -> int:
    """
    计算给定字符串所占用的 Token 数量。

    这个工具函数在系统的多个关键点被使用，主要服务于两个目的：
    1.  **成本控制**: 通过预先计算 Token 数量，可以在发送请求前就知道这次调用
        大约会消耗多少资源，有助于实现精确的成本控制。
    2.  **长度限制**: 帮助系统在发送请求前判断内容是否会超出模型的上下文长度限制，
        从而能够提前进行必要的截断或分段处理。

    Args:
        text (str): 需要计算 Token 的输入字符串。

    Returns:
        int: 字符串对应的 Token 数量。
    """
    return len(encoding.encode(text))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def execute_llm_call(
    messages: List[Dict[Literal["role", "content"], str]],
    config: LLMConfig
) -> Tuple[str, Dict[str, Any]]:
    """
    执行对大语言模型的异步调用。

    这是整个 LLM 工具库的核心函数，它封装了与 OpenAI API 交互的完整流程。
    该函数的设计充分考虑了实际生产环境中的各种挑战，集成了多个关键功能。

    核心功能:
    1.  **自动重试**: 使用 `tenacity` 库实现了智能的重试机制。
        - 采用指数退避策略（1-60秒之间随机等待）
        - 最多重试3次
        - 在网络问题、服务器临时错误等情况下自动触发
    2.  **Token 统计**: 提供详细的 Token 使用统计。
        - 请求前计算 prompt tokens
        - 从 API 响应中获取 completion tokens
        - 返回总 Token 使用量
    3.  **错误处理**: 实现了完整的异常捕获和处理流程。
        - 捕获所有可能的异常
        - 记录详细的错误日志
        - 返回用户友好的错误消息
    4.  **日志记录**: 在关键节点记录操作日志。
        - 可选的详细调试日志（完整提示和回复）
        - 错误日志包含异常详情
    5.  **类型安全**: 使用 Python 类型注解确保接口的正确使用。

    Args:
        messages (List[Dict]): 发送给 LLM 的消息列表，必须遵循 OpenAI 的标准格式：
            [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "用户输入"},
                ...
            ]
        config (LLMConfig): LLM 调用配置对象，包含：
            - model: 要使用的模型名称
            - temperature: 生成温度参数

    Returns:
        Tuple[str, Dict[str, Any]]: 一个元组，包含：
            - completion_text (str): LLM 返回的文本内容。如果调用失败，
              则返回一个友好的错误提示。
            - stats (Dict[str, Any]): 包含以下统计信息：
                - prompt_tokens: 提示部分的 Token 数量
                - completion_tokens: 回复部分的 Token 数量
                - total_tokens: 总 Token 使用量
                - model: 使用的模型名称
    """
    # 初始化统计数据字典
    stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": config.model}
    full_content = ""
    full_content = ""

    try:
        # 计算提示部分的 Token 数量
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt_tokens = count_tokens(prompt_text)
        stats["prompt_tokens"] = prompt_tokens

        # 调试日志，记录发送给 LLM 的完整提示（可根据需要启用）
        # logger.debug(f"发送给LLM的提示 (Tokens: {prompt_tokens}):\n---PROMPT START---\n{messages}\n---PROMPT END---")

        # 异步调用 OpenAI 的 ChatCompletion API
        response = await aclient.chat.completions.create(model=config.model,
        messages=messages,
        temperature=config.temperature)

        # 提取并清理返回的文本内容
        # 从 response 对象中安全地获取 content 属性，避免使用字典下标访问方式
        completion_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        # 提取 API 返回的 Token 使用情况
        usage = response.usage # type: ignore

        # 更新统计数据
        stats["completion_tokens"] = usage.completion_tokens
        stats["total_tokens"] = usage.total_tokens

        # 调试日志，记录 LLM 的回复
        # logger.debug(f"LLM回复 (Tokens: {stats['completion_tokens']}):\n---COMPLETION START---\n{completion_text}\n---COMPLETION END---")

        return completion_text, stats

    except Exception as e:
        # 如果在调用过程中发生任何异常，记录错误日志并返回一个安全的默认消息
        logger.exception(f"LLM调用失败: {e}")
        return "抱歉，我在思考时遇到了一点问题，请稍后再试。", stats

        



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def execute_llm_call_stream(
    messages: List[Dict[Literal["role", "content"], str]],
    config: LLMConfig,
    stream_callback: Optional[callable] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    执行对大语言模型的流式异步调用。

    这是支持流式响应的LLM调用函数，它在原有execute_llm_call基础上增加了流式处理能力。
    通过回调函数机制，可以实时接收和处理LLM生成的内容片段，从而实现更好的用户体验。

    核心功能:
    1.  **流式处理**: 启用OpenAI API的stream模式，实时接收生成内容
    2.  **实时回调**: 通过回调函数将内容片段实时推送给调用方
    3.  **完整兼容**: 保持与原有execute_llm_call相同的返回格式和错误处理
    4.  **自动重试**: 继承原有的重试机制和错误处理逻辑
    5.  **Token统计**: 提供完整的Token使用统计信息

    Args:
        messages (List[Dict]): 发送给 LLM 的消息列表，格式同execute_llm_call
        config (LLMConfig): LLM 调用配置对象
        stream_callback (Optional[callable]): 流式回调函数，接收参数：
            - content (str): 当前接收到的内容片段
            - status (str): 响应状态，'start', 'content', 'end', 'error'

    Returns:
        Tuple[str, Dict[str, Any]]: 与execute_llm_call相同的返回格式：
            - completion_text (str): 完整的LLM回复内容
            - stats (Dict[str, Any]): Token使用统计信息
    """
    # 初始化统计数据字典
    stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": config.model}
    full_content = ""
    
    try:
        # 计算提示部分的 Token 数量
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt_tokens = count_tokens(prompt_text)
        stats["prompt_tokens"] = prompt_tokens

        # 发送开始标志
        if stream_callback:
            try:
                print(f"发送流式开始标志")
                await stream_callback("", "start")
            except Exception as callback_error:
                logger.warning(f"流式开始回调函数执行失败: {callback_error}")

        # 异步调用 OpenAI 的 ChatCompletion API，启用流式模式
        stream = await aclient.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            stream=True  # 启用流式响应
        )

        # 处理流式响应
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                full_content += content_chunk
                
                # 如果提供了回调函数，则实时推送内容片段
                if stream_callback:
                    try:
                        # print(f"发送流失响应片段")
                        await stream_callback(content_chunk, "content")
                    except Exception as callback_error:
                        logger.warning(f"流式回调函数执行失败: {callback_error}")

        # 计算完成部分的Token数量
        completion_tokens = count_tokens(full_content)
        stats["completion_tokens"] = completion_tokens
        stats["total_tokens"] = prompt_tokens + completion_tokens

        # 发送最终完成回调
        if stream_callback:
            try:
                print(f"发送流失最终回调")
                await stream_callback("", "end")
            except Exception as callback_error:
                logger.warning(f"流式完成回调函数执行失败: {callback_error}")

        return full_content.strip(), stats

    except Exception as e:
        # 如果在调用过程中发生任何异常，记录错误日志并返回一个安全的默认消息
        logger.exception(f"LLM流式调用失败: {e}")
        
        # 如果有回调函数，也要通知错误状态
        if stream_callback:
            try:
                error_message = "抱歉，我在思考时遇到了一点问题，请稍后再试。"
                await stream_callback(error_message, "error")
            except Exception as callback_error:
                logger.warning(f"错误状态回调函数执行失败: {callback_error}")
        
        return "抱歉，我在思考时遇到了一点问题，请稍后再试。", stats
        