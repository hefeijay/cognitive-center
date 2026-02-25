# cognitive_model/tools/native/file_editor.py

import os
import logging
import json
from typing import Dict, Any

# 初始化日志记录器
logger = logging.getLogger("app.cognitive_model.tools.file_editor")

# --- [安全核心] 定义允许被修改的文件白名单 ---
# 为了防止 AI “越狱”并修改其认知边界之外的任意系统文件，我们在此定义一个严格的白名单。
# 只有在 `ALLOWED_FILES` 字典中列出的文件才允许被此工具访问。
# 使用 `os.path.abspath` 来将相对路径转换为绝对路径，确保路径匹配的唯一性和准确性。
COGNITIVE_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ALLOWED_FILES = {
    # 将逻辑键名映射到具体的、经过授权的文件路径
    "prompts_config": os.path.join(COGNITIVE_MODEL_PATH, 'config', 'prompts.json'),
    "system_constitution": os.path.join(COGNITIVE_MODEL_PATH, 'config', 'system_prompt.json'),
    "tools_config": os.path.join(COGNITIVE_MODEL_PATH, 'tools', 'tools.json')
}

def _get_whitelisted_path(file_key: str) -> str:
    """
    内部辅助函数：根据提供的逻辑键名，检查并获取其在白名单中对应的绝对文件路径。

    :param file_key: 文件的逻辑键名 (例如, "prompts_config")。
    :return: 如果键名有效，返回对应的绝对文件路径。
    :raises PermissionError: 如果提供的 `file_key` 不在 `ALLOWED_FILES` 白名单中，则抛出权限错误。
    """
    path = ALLOWED_FILES.get(file_key)
    if not path:
        raise PermissionError(f"文件键 '{file_key}' 不在允许的操作白名单中。这是一个安全限制。")
    return path

def read_file(file_key: str) -> str:
    """
    安全地读取一个在白名单内的配置文件的内容。

    :param file_key: 要读取的文件的逻辑键名 (例如, "prompts_config")。
    :return: 文件的内容字符串。如果发生错误（如权限问题、文件未找到），则返回格式化的错误信息字符串。
    """
    logger.info(f"请求读取白名单文件，键: '{file_key}'")
    try:
        # 首先通过辅助函数验证并获取合法的文件路径
        filepath = _get_whitelisted_path(file_key)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"成功读取文件: {filepath}")
        return content
    except (PermissionError, FileNotFoundError, IOError) as e:
        logger.error(f"读取文件 '{file_key}' 时出错: {e}")
        return f"错误：读取文件 '{file_key}' 失败: {str(e)}"

def write_file(file_key: str, new_content: str) -> str:
    """
    安全地覆写一个在白名单内的配置文件的全部内容。这是一个高权限操作。

    核心安全措施:
    -   **白名单检查**: 严格限制写入操作只能针对 `ALLOWED_FILES` 中定义的文件。
    -   **JSON格式校验**: 在写入前，会尝试将 `new_content` 解析为 JSON。如果解析失败，则拒绝写入，以防止破坏配置文件的结构完整性。

    :param file_key: 要写入的文件的逻辑键名 (例如, "prompts_config")。
    :param new_content: 要写入的全新内容字符串，必须是有效的 JSON 格式。
    :return: 一个描述操作结果的状态信息字符串。
    """
    logger.warning(f"高权限操作：请求写入文件，键: '{file_key}'")
    try:
        filepath = _get_whitelisted_path(file_key)
        
        # --- 安全校验：在写入前，确保新内容是有效的JSON格式 ---
        # 这是防止因格式错误而导致整个认知模型配置失效的关键保护层。
        try:
            json.loads(new_content)
        except json.JSONDecodeError as json_err:
            logger.error(f"写入被拒绝：提供给 '{file_key}' 的内容不是有效的JSON格式。错误: {json_err}")
            return f"错误：写入失败，因为提供的内容不是有效的JSON。请检查语法。"

        # 执行文件写入操作
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"成功写入文件: {filepath}")
        return f"文件 '{file_key}' 已成功更新。"
    except (PermissionError, IOError) as e:
        logger.error(f"写入文件 '{file_key}' 时出错: {e}")
        return f"错误：写入文件 '{file_key}' 失败: {str(e)}"