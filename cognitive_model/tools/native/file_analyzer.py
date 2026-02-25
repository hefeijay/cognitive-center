# cognitive_model/tools/native/file_analyzer.py (LLM驱动的智能分析版本)

import asyncio
import logging
import os
import json
from typing import Dict, Any
from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig

# 初始化日志记录器
logger = logging.getLogger("app.cognitive_model.tools.file_analyzer")

# 定义一个常量来限制读取的字符数，防止因文件过大导致 LLM 调用成本过高或超出上下文长度限制
MAX_CHARS_TO_READ = 12000  # 大约等于 3000-4000 tokens

def _build_analysis_prompt(file_content: str, filepath: str) -> str:
    """
    构建一个专门用于文件内容分析的 Prompt。
    这是一个内部辅助函数，旨在生成结构化、目标明确的指令，引导 LLM 完成分析任务。

    :param file_content: 要分析的文件内容字符串。
    :param filepath: 文件的原始路径，用于在 Prompt 中提供上下文。
    :return: 一个格式化的、准备好发送给 LLM 的 Prompt 字符串。
    """
    prompt = f"""
# 角色
你是一个高级文件分析助手。你的任务是阅读给定的文件内容，并以结构化的JSON格式返回你的分析报告。

# 任务
请分析以下来自文件 `{os.path.basename(filepath)}` 的内容，并提供：
1.  `summary`: 对文件整体内容的简洁摘要（不超过100字）。
2.  `key_points`: 一个包含3到5个关键信息点的字符串列表。
3.  `sentiment`: 对文件内容整体情绪的判断（例如："中性", "积极", "负面", "包含错误日志"）。

# 输出格式
你的回答必须是一个严格的JSON对象，不包含任何其他解释或文本。结构如下：
{{
    "summary": "...",
    "key_points": ["...", "...", "..."],
    "sentiment": "..."
}}

# 文件内容
---
{file_content}
---
"""
    return prompt

async def analyze_file(filepath: str) -> Dict[str, Any]:
    """
    一个智能异步工具，通过调用 LLM 来读取并分析一个本地文件。

    核心功能:
    -   **智能分析**: 利用 LLM 的理解能力，对文件内容进行摘要、关键点提取和情绪分析，而不仅仅是简单的文本处理。
    -   **异步执行**: 设计为异步函数，允许在等待 LLM 响应时非阻塞地执行其他任务。
    -   **安全截断**: 为了控制成本和避免超出模型限制，只读取文件的开头部分进行分析。

    业务逻辑:
    1.  检查文件是否存在，如果不存在则抛出 `FileNotFoundError`。
    2.  读取文件内容，并根据 `MAX_CHARS_TO_READ` 进行截断。如果文件被截断，则设置一个标志位。
    3.  调用 `_build_analysis_prompt` 构建针对该文件内容的定制化 Prompt。
    4.  准备消息列表，并配置 LLM（如使用 gpt-4o 模型）。
    5.  调用 `execute_llm_call` 异步执行对 LLM 的调用。
    6.  尝试将 LLM 返回的字符串响应解析为 JSON 对象。如果解析失败，则记录错误并返回原始文本。
    7.  将文件路径、分析结果、截断状态和 LLM 调用统计数据整合到一个字典中作为最终结果返回。
    8.  全面的异常处理确保在任何步骤失败时都能返回一个包含错误信息的字典。

    :param filepath: 要分析的文件的绝对路径。
    :return: 一个包含分析结果的字典。成功时包含 `file_path`, `analysis`, `is_truncated`, `llm_stats`；失败时包含 `error` 和 `message`。
    """
    logger.info(f"开始智能分析文件: '{filepath}'...")
    
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件未找到: {filepath}")

        # 读取文件内容，并进行截断处理
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(MAX_CHARS_TO_READ + 1)  # 多读一个字符来判断是否被截断
        
        is_truncated = False
        if len(content) > MAX_CHARS_TO_READ:
            content = content[:MAX_CHARS_TO_READ]
            is_truncated = True
            logger.warning(f"文件 '{filepath}' 内容过长，已截断为前 {MAX_CHARS_TO_READ} 个字符进行分析。")

        # 构建专用的分析Prompt
        analysis_prompt = _build_analysis_prompt(content, filepath)

        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": f"请分析文件: {os.path.basename(filepath)}"}
        ]
        
        # 调用LLM进行分析
        logger.info(f"正在调用LLM对文件 '{os.path.basename(filepath)}' 的内容进行分析...")
        config = LLMConfig(model="gpt-4o", temperature=0.6)
        llm_response_str, stats = await execute_llm_call(messages, config)
        
        # 尝试将LLM的响应解析为JSON
        try:
            analysis_result = json.loads(llm_response_str)
        except json.JSONDecodeError:
            logger.error("LLM未能返回有效的JSON格式分析报告。将返回原始文本。")
            analysis_result = {"error": "LLM response was not valid JSON", "raw_response": llm_response_str}
        
        # 将所有信息整合到最终结果中
        final_result = {
            "file_path": filepath,
            "analysis": analysis_result,
            "is_truncated": is_truncated,
            "llm_stats": stats
        }
        
        logger.info(f"文件 '{os.path.basename(filepath)}' 智能分析完成。")
        return final_result

    except Exception as e:
        logger.error(f"分析文件 '{filepath}' 时出错: {e}")
        return {
            "error": True,
            "message": str(e)
        }