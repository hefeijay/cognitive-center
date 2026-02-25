# cognitive_model/tools/tool_registry.py (v4.0 - 数据库驱动版)

import logging
import importlib
from typing import Dict, Any, Optional, List
import asyncio
import requests
import json
import traceback
from urllib.parse import urlencode, quote
import sseclient
import ast
from sqlalchemy.orm import Session
from db_models.db_session import db_session_factory
from singa_one_server.repositories.tool_repository import ToolRepository
from db_models.tool import Tool

logger = logging.getLogger("app.cognitive_model.tools.tool_registry")

class ToolRegistry:
    """
        动态工具注册与执行中心 (v4.0 - 数据库驱动版)

        核心职责:
        1.  **数据库加载**: 从数据库加载所有工具的定义。
        2.  **按需实例化**: 提供 `get_tools_by_ids` 方法，允许根据会话或请求动态加载和实例化特定的自定义工具。
        3.  **元数据提供**: 提供当前激活工具集的详细描述，用于构建路由 Agent 的 Prompt。
        4.  **统一执行**: 提供一个统一的接口 `execute_tool`，用于执行所有类型的工具。
    """
    def __init__(self):
        """
            初始化工具注册表。
            - 初始化工具仓库。
            - 从数据库加载所有工具。
        """
        self.tool_repository = ToolRepository()
        self._tools_by_id: Dict[str, Tool] = {}
        self._tools_by_name: Dict[str, Tool] = {}
        self.load_tools()

    def load_tools(self):
        """
            从数据库中加载所有工具。
        """
        try:
            # logger.info("正在从数据库加载所有工具...")
            with db_session_factory() as db:
                all_tools = self.tool_repository.get_all_tools(db)
                self._tools_by_id = {tool.tool_id: tool for tool in all_tools}
                self._tools_by_name = {tool.name: tool for tool in all_tools}
                logger.info(f"'工具'加载成功!")
                # print(f"工具信息 (by ID): {self._tools_by_id}")
        except Exception as e:
            logger.exception(f"从数据库加载工具时发生未知错误: {e}")
            self._tools_by_id = {}
            self._tools_by_name = {}

    def get_tools_by_ids(self, tool_ids: list[str]) -> Dict[str, Tool]:
        """
        根据提供的工具ID列表，从已加载的工具中查找并返回一个包含相应工具实例的字典。
        
        :param tool_ids: 一个包含工具ID的字符串列表。
        :return: 一个以工具ID为键，工具对象为值的字典。
        """
        if not tool_ids:
            return {}
        
        logger.debug(f"请求的工具ID列表: {tool_ids}")
        logger.debug(f"已加载的工具 (by ID): {list(self._tools_by_id.keys())}")

        custom_tools = {
            tool_id: self._tools_by_id[tool_id]
            for tool_id in tool_ids
            if tool_id in self._tools_by_id
        }
        
        missing_tools = [tool_id for tool_id in tool_ids if tool_id not in self._tools_by_id]
        if missing_tools:
            logger.warning(f"请求的工具中，有 {len(missing_tools)} 个未在定义中找到: {missing_tools}")
            
        return custom_tools

    def get_all_tool_descriptions(self, tools: Optional[Dict[str, Tool]] = None) -> str:
        """
            动态生成指定工具集的格式化描述文本。
            如果未提供工具集，则默认使用当前激活的工具（self._tools_by_name）。

            :param tools: 一个可选的工具字典。如果提供，则为该字典中的工具生成描述。
            :return: 一个包含所有工具描述的字符串。
        """
        # 如果传入的是按ID索引的字典，需要先转换为按名称索引
        if tools and tools.keys() and list(tools.keys())[0] in self._tools_by_id:
             tool_source = {tool.name: tool for tool in tools.values()}
        else:
             tool_source = tools if tools is not None else self._tools_by_name

        if not tool_source:
            return "当前没有可用的工具。"
        
        descriptions = []
        for name, tool in tool_source.items():
            try:
                schema = json.loads(tool.schema_def) if isinstance(tool.schema_def, str) else tool.schema_def
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"工具 '{name}' 的 schema_def 格式不正确，已跳过。内容: {tool.schema_def}")
                schema = {}

            params = schema.get("properties", {})
            param_desc_list = [f"{p_name} ({p_info.get('description', '无描述')})" for p_name, p_info in params.items()]
            param_desc = ", ".join(param_desc_list)
            
            descriptions.append(
                f"- 工具名称: `{name}`\n"
                f"  功能描述: {tool.description}\n"
                f"  执行模式: {tool.mode}\n"
                f"  参数: {param_desc if param_desc else '无'}"
            )
        return "\n".join(descriptions)
    
    def get_tool_info(self, name: str, tool_set: Optional[Dict[str, Tool]] = None) -> Optional[Tool]:
        """
            根据名称从指定的工具集中获取工具的完整定义元数据。
            如果未提供工具集，则从默认的 `_tools_by_name` 中查找。

            :param name: 工具的名称。
            :param tool_set: 从中查找工具的可选字典。
            :return: 工具对象，如果找不到则返回 None。
        """
        source = tool_set if tool_set is not None else self._tools_by_name
        return source.get(name)

    def execute_tool(self, name: str, args: Dict[str, Any], tool_set: Optional[Dict[str, Tool]] = None) -> str:
        """
            根据名称和参数，从指定的工具集中执行一个工具。
            这是工具系统的核心入口点。

            :param name: 要执行的工具名称。
            :param args: 一个包含参数名和值的字典。
            :param tool_set: 执行时使用的临时工具集。如果为 None，则使用默认工具集。
            :return: 工具执行结果的字符串形式。
        """
        tool = self.get_tool_info(name, tool_set)
        if not tool:
            # 如果在临时工具集中找不到，也检查一下默认工具集
            tool = self.get_tool_info(name)
            if not tool:
                error_msg = f"错误：尝试执行的工具 '{name}' 未在任何可用工具集中找到。"
                logger.error(error_msg)
                return error_msg

        logger.info(f"准备执行工具 '{name}' (类型: {tool.type})，参数: {args}")

        try:
            # 根据工具类型，分发到不同的执行逻辑
            if tool.type == "internal_python":
                return self._execute_internal_python(tool, args)
            elif tool.type == "external_api":
                return self._execute_external_api(tool, args)
            else:
                raise NotImplementedError(f"未知的工具类型: '{tool.type}'")
        except Exception as e:
            logger.exception(f"执行工具 '{name}' 时发生严重错误: {e}")
            return f"错误：在执行工具 '{name}' 时发生了内部故障: {str(e)}"

    def _execute_internal_python(self, tool: Tool, args: Dict[str, Any]) -> str:
        """
            执行一个内部 Python 函数类型的工具。
            使用 `importlib` 动态加载模块和函数，并支持同步和异步调用。

            :param tool: 工具对象。
            :param args: 工具执行所需的参数。
            :return: 函数执行结果的字符串表示。
            :raises ValueError: 如果工具配置不完整或无法找到指定的模块/函数。
        """
        location = tool.location
        module_path, function_name = location.get("module"), location.get("function")
        if not module_path or not function_name:
            raise ValueError(f"工具 '{tool.name}' 的 'location' 配置不完整。")
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            # 获取函数对象
            func_to_call = getattr(module, function_name)
            
            # 判断函数是同步还是异步，并以相应方式调用
            if asyncio.iscoroutinefunction(func_to_call):
                 # 为异步函数创建一个新的事件循环来运行它
                 # 注意：这是一个简化的同步包装器。在已有的异步环境中，应使用 `await`。
                 return asyncio.run(func_to_call(**args))
            else:
                 # 直接调用同步函数
                 return str(func_to_call(**args))
        except (ImportError, AttributeError) as e:
            raise ValueError(f"无法找到工具 '{tool.name}' 的执行函数。请检查配置: {e}")

    def _execute_external_api(self, tool: Tool, args: Dict[str, Any]) -> str:
        """
        执行一个外部 API 类型的工具。
        支持标准的 request-response 和 SSE (Server-Sent Events) 流式 API。

        :param tool: 工具对象。
        :param args: 工具执行所需的参数，将作为请求的 body 或查询参数。
        :return: API 响应结果的字符串表示。
        :raises ValueError: 如果 API URL 未配置。
        :raises NotImplementedError: 如果配置了不支持的 HTTP 方法。
        :raises ConnectionError: 如果网络请求失败。
        """
        try:
            location_str = tool.location
            location = json.loads(location_str) if isinstance(location_str, str) else location_str
        except (json.JSONDecodeError, TypeError):
            logger.error(f"工具 '{tool.name}' 的 location 格式不正确: {tool.location}")
            raise ValueError(f"工具 '{tool.name}' 的 location 配置无效。")

        if location.get("stream_api"):
            return self._execute_sse_api(tool, args, location)
        else:
            return self._execute_standard_api(tool, args, location)

    def _execute_standard_api(self, tool: Tool, args: Dict[str, Any], location: Dict[str, Any]) -> str:
        """
        执行标准的 request-response 模式的外部 API。
        """
        api_url = location.get("url")
        method = location.get("method", "POST").upper()
        headers = location.get("headers")

        if not api_url:
            raise ValueError(f"外部API工具 '{tool.name}' 的'url'未配置。")

        timeout_seconds = 300
        logger.info(f"执行一个外部API请求: {method} {api_url}, 参数: {args}, Headers: {headers}")

        try:
            if method == "POST":
                response = requests.post(api_url, json=args, headers=headers, timeout=timeout_seconds)
            elif method == "GET":
                response = requests.get(api_url, params=args, headers=headers, timeout=timeout_seconds)
            else:
                raise NotImplementedError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()

            response_json = response.json()
            logger.info(f"收到外部API响应: {response_json}")

            if "result" in response_json:
                return str(response_json["result"])
            elif "error" in response_json:
                logger.warning(f"外部API工具 '{tool.name}' 返回业务错误: {response_json['error']}")
                return f"外部工具API返回错误: {response_json['error']}"
            else:
                return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"调用外部API工具 '{tool.name}' 失败: {e}")
            raise ConnectionError(f"调用外部API失败: {e}")
    def _convert_bools(self, obj):
        """递归将 dict/list 中的布尔值转为 'True'/'False' 字符串"""
        if isinstance(obj, bool):
            return "True" if obj else "False"
        elif isinstance(obj, dict):
            return {k: self._convert_bools(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_bools(i) for i in obj]
        return obj

    def _normalize_sse_params(self,args: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化 SSE 请求参数：
        1. 顶层布尔值转为 "True"/"False"
        2. config 内部布尔值也转为 "True"/"False"
        3. 其他复合类型统一 JSON 序列化
        """
        normalized: Dict[str, Any] = {}
        for k, v in (args or {}).items():
            if k == "config":
                try:
                    if isinstance(v, dict):
                        normalized[k] = json.dumps(self._convert_bools(v), ensure_ascii=False)
                    elif isinstance(v, str):
                        cfg_obj = None
                        try:
                            cfg_obj = json.loads(v)
                        except Exception:
                            try:
                                cfg_obj = ast.literal_eval(v)
                            except Exception:
                                cfg_obj = None
                        if isinstance(cfg_obj, (dict, list)):
                            normalized[k] = json.dumps(self._convert_bools(cfg_obj), ensure_ascii=False)
                        else:
                            normalized[k] = v
                    else:
                        normalized[k] = json.dumps(self._convert_bools(v), ensure_ascii=False)
                except Exception:
                    normalized[k] = v
                continue

            # 顶层布尔值 → "True"/"False"
            if isinstance(v, bool):
                normalized[k] = "True" if v else "False"
            elif isinstance(v, (dict, list)):
                try:
                    normalized[k] = json.dumps(self._convert_bools(v), ensure_ascii=False)
                except Exception:
                    normalized[k] = str(v)
            else:
                normalized[k] = str(v)

        return normalized

    def _execute_sse_api(self, tool: Tool, args: Dict[str, Any], location: Dict[str, Any], on_chunk: Optional[callable] = None) -> str:
        """
        通过 SSE (Server-Sent Events) 执行外部 API 调用。

        业务逻辑:
        1.  从 `location` 中获取 SSE API 的 URL。
        2.  构建完整的请求 URL，将 `args` 作为查询参数。
        3.  使用 `sseclient` 发起 GET 请求并处理 SSE 事件流。
        4.  监听 `message` 事件，累积从事件流中接收到的数据。
        5.  在事件流结束时，返回累积的完整数据。
        6.  处理网络连接错误和异常，并记录日志。
        """
        sse_url = location.get("url") 
        if not sse_url:
            raise ValueError("工具的 location 配置中缺少 'url' 字段。")

        # 统一规范化参数（包含 config 及布尔值处理）
        params = self._normalize_sse_params(args)
        headers = {"Accept": "text/event-stream"}
        # 提高 SSE 超时，避免服务端首事件延迟导致超时
        timeout = 120

        logger.debug(f"SSE request URL: {sse_url}")
        logger.debug(f"SSE request params: {params}")
        print(f"SSE request URL: {sse_url}")
        # 使用ensure_ascii=False确保中文字符正确显示,indent=2使输出格式更易读
        print(f"SSE request params: {json.dumps(params, ensure_ascii=False, indent=2)}")
        try:
            response = requests.get(sse_url, params=params, headers=headers, stream=True, timeout=timeout)
            client = sseclient.SSEClient(response)
            print(f"sse请求{response}")
            accumulated_content = ""
            
            # 兼容旧版可迭代 / 新版需要 events()
            if hasattr(client, "events"):
                events_iter = client.events()  
                print("新版")
            else:
                events_iter = client     
                print("旧版")    

            for event in events_iter:

                logger.info(f"收到事件: {event}")
                if event.event == 'message':
                    try:
                        payload = json.loads(event.data)
                        print(payload)
                        inner = payload.get("data", {})
                        status = inner.get("status")

                        # 新版服务采用 status="stream"，文本位于顶层 content 字段
                        if status == "stream":
                            content_text = payload.get("content", "")
                            if isinstance(content_text, str) and content_text:
                                accumulated_content += content_text
                                if on_chunk:
                                    try:
                                        on_chunk(content_text, phase="stream")
                                    except Exception as cb_err:
                                        logger.warning(f"on_chunk 回调处理 stream 片段失败: {cb_err}")
                            continue

                        if status == "started":
                            logger.info("任务开始执行")
                        elif status == "processing":
                            logger.info("任务处理中")
                            # 处理处理中阶段的代理增量响应（旧版协议）
                            response = inner.get("response", {})
                            agent_response = response.get("agent_response", "")
                            if agent_response:
                                accumulated_content += agent_response
                                # 将增量内容通过桥接回调推送到前端
                                if on_chunk:
                                    try:
                                        on_chunk(agent_response, phase="processing")
                                    except Exception as cb_err:
                                        logger.warning(f"on_chunk 回调处理 processing 片段失败: {cb_err}")
                        elif status == "completed":
                            logger.info("任务完成，关闭 SSE 连接")
                            # 处理完成阶段的最终答案（兼容新版与旧版协议）
                            answer = inner.get("answer")
                            # 新版可能将最终答案放在顶层 answer 或 content
                            if not answer:
                                answer = payload.get("answer")
                            if not answer:
                                answer = payload.get("content")

                            if isinstance(answer, str) and answer:
                                accumulated_content += answer
                                if on_chunk:
                                    try:
                                        on_chunk(answer, phase="completed")
                                    except Exception as cb_err:
                                        logger.warning(f"on_chunk 回调处理 completed 片段失败: {cb_err}")
                            else:
                                # 某些实现可能将最终文本仍放在 response.agent_response
                                response = inner.get("response", {})
                                agent_response = response.get("agent_response", "")
                                if agent_response:
                                    accumulated_content += agent_response
                                    if on_chunk:
                                        try:
                                            on_chunk(agent_response, phase="completed")
                                        except Exception as cb_err:
                                            logger.warning(f"on_chunk 回调处理 completed(agent_response) 片段失败: {cb_err}")
                            break
                        else:
                            # 未知状态，尝试提取顶层 content，否则按纯文本处理
                            fallback_text = payload.get("content")
                            if isinstance(fallback_text, str) and fallback_text:
                                accumulated_content += fallback_text
                                if on_chunk:
                                    try:
                                        on_chunk(fallback_text, phase="unknown")
                                    except Exception as cb_err:
                                        logger.warning(f"on_chunk 回调处理 unknown(content) 片段失败: {cb_err}")
                            else:
                                accumulated_content += event.data
                                if on_chunk:
                                    try:
                                        on_chunk(event.data, phase="unknown")
                                    except Exception as cb_err:
                                        logger.warning(f"on_chunk 回调处理 unknown 片段失败: {cb_err}")

                    except json.JSONDecodeError:
                        accumulated_content += event.data
                        if on_chunk:
                            try:
                                on_chunk(event.data, phase="raw")
                            except Exception as cb_err:
                                logger.warning(f"on_chunk 回调处理 raw 片段失败: {cb_err}")
            
            logger.info(f"SSE stream completed. Accumulated content length: {len(accumulated_content)}")
            return accumulated_content

        except requests.exceptions.RequestException as e:
            logger.error(f"调用 SSE API 工具 '{tool.name}' 时发生网络错误: {e}", exc_info=True)
            raise ConnectionError(f"无法连接到 SSE 服务 at {sse_url}。") from e
        except Exception as e:
            logger.error(f"执行工具 '{tool.name}' 时发生严重错误: {e}", exc_info=True)
            raise ConnectionError(f"无法连接到 SSE 服务 at {sse_url}。") from e


# tool_registry = ToolRegistry()
