# cognitive_model/handlers/query_handler.py

import time
import uuid
import logging, asyncio, re, json
from typing import Dict, Any, Optional, List

from flask_migrate import history
from singa_one_server.repositories import chat_history_repository
from cognitive_model.agents.llm_utils import execute_llm_call, LLMConfig, format_messages_for_llm
from .base_handler import BaseHandler

logger = logging.getLogger("app.handler.query")

class QueryHandler(BaseHandler):
    """
    处理“提问”和“反馈”意图。
    这是最核心和复杂的处理器，负责：
    - 路由决策（工具使用或直接回答）
    - 记忆检索与管理
    - 工具的同步/异步执行
    - 调用思考智能体生成答案
    - 对记忆进行总结和更新
    """
    async def handle_stream(self, orchestrator, user_input: str, session_id: str, stream_callback=None, **kwargs) -> Dict[str, Any]:
        """
        处理用户查询的流式版本，支持实时流式回复。

        这是QueryHandler的流式版本，它在原有功能基础上增加了实时流式响应能力。
        主要流程与handle方法相同，但在调用ThinkingAgent时使用流式版本。

        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入字符串。
            session_id: 当前的会话ID。
            stream_callback: 流式回调函数，用于实时推送内容片段。
            **kwargs: 其他可能需要的参数，如意图（intent）。

        Returns:
            一个包含最终响应和完整助手响应的字典。
        """
        # 1. 分类和加载记忆
        route_decision, master_topic, memory_node = await self._classify_and_load_memory(
            orchestrator, user_input, session_id, orchestrator.tool_config, orchestrator.model_config
        )
        # [强制路由] 不论原始决策如何，统一改为调用日本路上养殖工具
        # 通过覆盖 route_decision，确保后续流程以工具调用路径执行。
        # 传入 user_input 作为 query，agent_type 固定为 "japan"
        # 将上游传入的 message_id（若有）一并传递到工具决策中
        route_decision = self._force_japan_aquaculture_decision(
            route_decision, session_id, user_input, agent_type="japan", message_id=kwargs.get("message_id")
        )
        topic_summary = memory_node.get("topic_content", "") if memory_node else ""

        print(f"用户:{user_input}")
        # 2. 执行决策路径（流式版本）
        decision_result = await self._execute_decision_path_stream(
            orchestrator, route_decision, user_input, master_topic, session_id, stream_callback
        )

        # 3. 准备最终响应
        return self._prepare_response(
            kwargs.get("intent", "提问"),
            decision_result["final_user_facing_content"],
            decision_result["full_assistant_response"],
            decision_result["should_synthesize_speech"]
        )

    async def _execute_decision_path_stream(self, orchestrator, route_decision: dict, user_input: str, master_topic: str, session_id: str, stream_callback=None) -> dict:
        """
        执行决策路径的流式版本。

        根据路由决策，选择是使用工具还是直接生成回答，支持流式响应。

        Args:
            orchestrator: 认知协调器实例。
            route_decision: 路由决策结果。
            user_input: 用户输入。
            master_topic: 主题。
            session_id: 会话ID。
            stream_callback: 流式回调函数。

        Returns:
            包含决策执行结果的字典。
        """
        if route_decision.get("use_tool", False):
            # 工具使用（流式）：为异步工具路径注入统一的流式回调，使片段按同一 message_id 聚合
            final_user_facing_content = await self._handle_async_tool_use_stream(
                orchestrator, session_id, master_topic, route_decision, stream_callback
            )
            return {
                "final_user_facing_content": final_user_facing_content,
                # 由 orchestrator.process_input_stream 外层的 stream_callback 累积全文，这里不回传以避免覆盖
                "full_assistant_response": "",
                "should_synthesize_speech": False,
                "is_async_task": True,
                "tool_result": None,
                "tool_name": route_decision.get("tool")
            }
        else:
            # 直接回答 - 使用流式版本
            final_user_facing_content, full_assistant_response = await self._invoke_thinking_agent_stream(
                orchestrator, user_input, stream_callback
            )
            
            return {
                "final_user_facing_content": final_user_facing_content,
                "full_assistant_response": full_assistant_response,
                "should_synthesize_speech": True,
                "is_async_task": False
            }

    async def _invoke_thinking_agent_stream(self, orchestrator, user_input: str, stream_callback=None) -> tuple[str, str]:
        """
        调用思考智能体生成流式回复。

        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的输入。
            stream_callback: 流式回调函数。

        Returns:
            一个元组，包含：
            - final_user_facing_content (str): 解析后最终面向用户的内容。
            - full_assistant_response (str): LLM返回的完整响应。
        """
        full_assistant_response, thinking_stats = await orchestrator.thinking_agent.run_stream(
            user_input, orchestrator.history, stream_callback
        )
        orchestrator._update_total_stats("thinking_agent", thinking_stats)
        final_user_facing_content = self._parse_final_answer(full_assistant_response) or full_assistant_response
        return final_user_facing_content, full_assistant_response

    async def handle(self, orchestrator, user_input: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        处理用户查询的核心方法，通过协调一系列私有方法来完成任务。

        此方法将复杂的处理流程分解为几个独立的步骤：
        1.  **分类和加载记忆**: 调用 `_classify_and_load_memory` 来确定主题并加载相关记忆和历史。
        2.  **执行决策路径**: 调用 `_execute_decision_path` 来执行工具或直接生成回答。
        3.  **更新记忆**: 对于非异步任务，调用 `_update_memory_and_summary` 来保存交互并更新摘要。
        4.  **准备响应**: 调用 `_prepare_response` 构建最终的响应对象。

        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入字符串。
            history: (在此实现中未使用，但保留以兼容接口)
            session_id: 当前的会话ID。
            session_tools: 本次请求专用的、动态加载的工具集。
            **kwargs: 其他可能需要的参数，如意图（intent）。

        Returns:
            一个包含最终响应和完整助手响应的字典。
        """
        # logger.info(f"[会话 {session_id}] 使用QueryHandler处理用户输入，携带 {len(orchestrator.tool_config)} 个工具...")

        # 1. 分类和加载记忆
        route_decision, master_topic, memory_node = await self._classify_and_load_memory(
            orchestrator, user_input, session_id, orchestrator.tool_config, orchestrator.model_config
        )
        # [强制路由] 不论原始决策如何，统一改为调用日本路上养殖工具
        # 传入 user_input 作为 query，agent_type 固定为 "japan"
        route_decision = self._force_japan_aquaculture_decision(
            route_decision, session_id, user_input, agent_type="japan", message_id=kwargs.get("message_id")
        )
        topic_summary = memory_node.get("topic_content", "") if memory_node else ""

        # TODO 是否需要记忆总结
        
        # 2. 执行决策路径
        decision_result = await self._execute_decision_path(
            orchestrator, route_decision, user_input, master_topic, session_id
        )

        # # 3. 更新记忆
        # if not decision_result["is_async_task"]:
        #     await self._update_memory_and_summary(
        #         orchestrator,
        #         memory_node,
        #         topic_history,
        #         user_input,
        #         decision_result["full_assistant_response"],
        #         decision_result.get("tool_result"),
        #         decision_result.get("tool_name")
        #     )
        
        # 4. 准备最终响应
        return self._prepare_response(
            kwargs.get("intent", "提问"),
            decision_result["final_user_facing_content"],
            decision_result["full_assistant_response"],
            decision_result["should_synthesize_speech"]
        )

    def _force_japan_aquaculture_decision(self, original_decision: dict, session_id: str, user_input: str, agent_type: str = "japan", message_id: str | None = None) -> dict:
        """
        强制路由到“日本路上养殖”工具的辅助方法。

        设计意图：
        - 不论上游路由智能体的决策为何，统一改为工具使用路径（tool_use）。
        - 使用异步模式（async），以匹配 SSE 工具的调用方式，并复用现有的参数覆盖逻辑。
        - 同时填充 `use_tool` 字段以兼容流式路径的判断逻辑。

        输入参数：
        - original_decision (dict): 上游路由返回的原始决策字典，可能包含 tool/args/mode 等。
        - session_id (str): 当前会话 ID，用于在同步工具路径中通知消息保存时传递。
        - user_input (str): 当前用户的原始问题，将作为 SSE 的 `query` 参数。
        - agent_type (str): 代理类型，SSE 侧用于路由不同专家，默认为 "japan"。

        输出结果：
        - dict: 覆盖后的决策字典，结构如下：
          {
            "route": "tool_use",
            "use_tool": True,
            "tool": "japan_aquaculture_expert",
            "mode": "async",
            "args": {"query": user_input, "agent_type": agent_type, "session_id": session_id, ...},
            "session_id": session_id
          }

        关键算法逻辑：
        - 若上游已有 args，则在不丢失其内容的基础上补充必需参数：`query`、`agent_type`、`session_id`。
        - `query` 优先使用当前的 `user_input`，以减少服务端从历史检索造成的首事件延迟。
        - 同步在 args 层面附带 `session_id`，确保 SSE 查询参数完整。
        """
        forced_args = dict((original_decision or {}).get("args", {}))
        # 注入必需的参数
        if not forced_args.get("query"):
            forced_args["query"] = user_input
        if not forced_args.get("agent_type"):
            forced_args["agent_type"] = agent_type
        # 在 args 层面也携带 session_id，避免遗漏
        forced_args["session_id"] = session_id
        # 若客户端提供 message_id，则透传给工具请求参数，确保端到端一致性
        if message_id:
            forced_args["message_id"] = message_id

        return {
            "route": "tool_use",
            "use_tool": True,
            "tool": "japan_aquaculture_expert",
            "mode": "async",
            "args": forced_args,
            "session_id": session_id,
        }

    def _parse_final_answer(self, full_response: str) -> str:
        """
        从思考智能体的完整响应中提取最终要呈现给用户的内容。
        LLM的输出可能包含多个部分（如思考过程、最终答案等），此函数旨在精确提取“答案”部分。
        
        业务逻辑:
        1.  查找 "【答案输出】" 标签作为答案开始的标记。
        2.  如果找到，则提取该标签之后、下一个Markdown列表项之前的所有内容作为最终答案。
        3.  如果未找到标签，则进行一次简单的清理，移除所有 "【...】" 格式的标签，并返回剩余部分。
        4.  如果解析后内容为空，则返回原始响应，便于调试。
        """
        stripped_response = full_response.strip()
        lines = [line.strip() for line in stripped_response.split('\n')]
        try:
            # 寻找答案输出的起始行
            start_index = next(i for i, line in enumerate(lines) if "【答案输出】" in line)
        except StopIteration:
            logger.warning("在LLM响应中未找到'【答案输出】'标签，将返回原始响应。")
            # 如果没有找到标签，尝试移除所有标签后返回
            return re.sub(r'【[^】]+】：\s*', '', stripped_response).strip() if "【" in stripped_response else stripped_response
        
        answer_lines = []
        for i in range(start_index, len(lines)):
            line = lines[i]
            if i == start_index:
                # 提取第一行中标签后的内容
                content_part = line.split('】：', 1)[-1].strip()
                if content_part:
                    answer_lines.append(content_part)
                continue
            # 遇到下一个标签时停止
            if line.startswith("- 【"):
                break
            answer_lines.append(line)
            
        final_answer = "\n".join(answer_lines).strip()
        if final_answer:
            logger.info("成功从Markdown列表结构中精确解析出'【答案输出】'。")
            print(f"解析后的答案：{final_answer}")
            return final_answer
        else:
            logger.warning("解析'【答案输出】'后内容为空，将返回原始响应以供调试。")
            return stripped_response

    async def _invoke_thinking_agent(self, orchestrator, user_input: str) -> tuple[str, str]:
        """
        调用思考智能体生成回复。

        Args:
            orchestrator: 认知协调器实例。
            system_prompt: 构建好的系统提示。
            user_input: 用户的输入。

        Returns:
            一个元组，包含：
            - final_user_facing_content (str): 解析后最终面向用户的内容。
            - full_assistant_response (str): LLM返回的完整响应。
        """
        full_assistant_response, thinking_stats = await orchestrator.thinking_agent.run(user_input, orchestrator.history)
        print(f"思考响应：{full_assistant_response}")
        orchestrator._update_total_stats("thinking_agent", thinking_stats)
        final_user_facing_content = self._parse_final_answer(full_assistant_response) or full_assistant_response
        return final_user_facing_content, full_assistant_response

    async def _execute_decision_path(self, orchestrator, route_decision: dict, user_input: str, master_topic: str, session_id: str) -> dict:
        """
            根据路由决策执行相应的处理路径（工具使用或直接回答）。

            Args:
                orchestrator: 认知协调器实例。
                route_decision: 路由决策结果。
                user_input: 用户输入。
                master_topic: 主题名称。
                topic_summary: 主题摘要。
                topic_history: 主题历史。
                full_session_history: 完整会话历史。
                session_id: 会话 ID。

            Returns:
                一个包含处理结果的字典。
        """
        final_user_facing_content = ""
        full_assistant_response = ""
        is_async_task = False
        should_synthesize_speech = True
        tool_result = None

        if route_decision.get("route") == "tool_use":
            tool_mode = route_decision.get("mode")
            if tool_mode == 'sync':
                logger.warning(f"[会话 {session_id}] 路径决策: 同步慢思考 (使用工具)")
                tool_result = await self._handle_sync_tool_use(orchestrator, master_topic, route_decision)
                final_user_facing_content, full_assistant_response = await self._invoke_thinking_agent(orchestrator, user_input)
            else:  # async
                logger.warning(f"[会话 {session_id}] 路径决策: 异步慢思考 (使用工具)")
                is_async_task = True
                should_synthesize_speech = False
                final_user_facing_content = await self._handle_async_tool_use(orchestrator, session_id, master_topic, route_decision)
                full_assistant_response = final_user_facing_content
        else: # direct_answer
            logger.info(f"[会话 {session_id}] 路径决策: 快思考 (直接回答)")
            final_user_facing_content, full_assistant_response = await self._invoke_thinking_agent(orchestrator,user_input)

        return {
            "final_user_facing_content": final_user_facing_content,
            "full_assistant_response": full_assistant_response,
            "is_async_task": is_async_task,
            "should_synthesize_speech": should_synthesize_speech,
            "tool_result": tool_result,
            "tool_name": route_decision.get("tool")
        }
        
    async def _classify_and_load_memory(self, orchestrator, user_input: str, session_id: str, session_tools: Dict[str, Any], model_config: dict):
        """
            对用户输入进行主题分类，并加载或创建相关的记忆节点和历史记录。

            Args:
                orchestrator: 认知协调器实例。
                user_input: 用户的原始输入。
                session_id: 当前会话ID。

            Returns:
                一个元组，包含：
                - route_decision (dict): 路由决策结果。
                - master_topic (str): 主题名称。
                - memory_node (dict): 加载的记忆节点。
                - topic_history (list): 主题相关的历史记录。
                - full_session_history (list): 完整的会话历史。
        """
        # 路由决策，判断是否需要使用工具
        route_decision, route_stats = await orchestrator.routing_agent.run(user_input, None, session_tools, model_config) # history暂不使用
        orchestrator._update_total_stats("routing_agent", route_stats)
        
        # 对用户输入进行主题分类
        candidate_topic, classification_stats = await orchestrator.classification_agent.run(user_input, model_config)
        orchestrator._update_total_stats("classification_agent", classification_stats)
        
        # 查找或创建主主题
        master_topic = await orchestrator.hippocampus.find_master_topic(candidate_topic)
        
        # 加载与主主题相关的记忆节点
        memory_node = await orchestrator.hippocampus.load_memory_node_by_master_topic(master_topic, session_id)
        
        return route_decision, master_topic, memory_node

    async def _update_memory_and_summary(
            self, 
            orchestrator, 
            memory_node: dict, 
            history: list,
            user_input: str, 
            full_assistant_response: str, 
            tool_result: str = None, 
            tool_name: str = None
        ):
        """
        更新主题记忆和摘要。

        此方法负责在一次交互结束后，将最新的对话内容（用户输入、AI回复、工具结果）
        追加到历史记录中，然后调用总结智能体生成新的摘要，并最终将更新后的记忆节点
        持久化到存储中。

        Args:
            orchestrator: 认知协调器实例。
            memory_node: 当前的记忆节点，包含主题信息但不包含历史记录。
            history: 当前主题的对话历史记录列表。
            user_input: 用户输入。
            full_assistant_response: AI的完整响应。
            tool_result: 工具执行结果，可选。
            tool_name: 使用的工具名称，可选。
        """
        # 1. 将最新的交互添加到历史记录中
        if history:
            formatted_history = [{"role": msg.role, "content": msg.content} for msg in history]
        formatted_history.append({"role": "user", "content": user_input})
        
        if tool_result is not None and tool_name is not None:
            tool_info_for_history = f"[同步工具 '{tool_name}' 已执行，结果: {tool_result}]"
            formatted_history.append({"role": "assistant", "content": tool_info_for_history})
        
        formatted_history.append({"role": "assistant", "content": full_assistant_response})

        # 2. 调用总结代理生成新的摘要
        new_summary, summarization_stats = await orchestrator.summarization_agent.summarize(
            memory_node.get("topic_content", ""),  # 使用 topic_content 作为当前摘要
            formatted_history
        )
        orchestrator._update_total_stats("summarization_agent", summarization_stats)

        # 3. 更新记忆节点并持久化
        memory_node["topic_content"] = new_summary  
        await orchestrator.hippocampus.update_memory_node(memory_node)
        
        # 4. 更新历史记录
        await orchestrator.hippocampus.update_history(memory_node["topic_id"], formatted_history)

    def _prepare_response(self, intent: str, final_user_facing_content: str, full_assistant_response: str, should_synthesize_speech: bool) -> dict:
        """
        准备最终返回给编排器的响应。

        Args:
            intent: 意图。
            final_user_facing_content: 最终面向用户的内容。
            full_assistant_response: AI的完整响应。
            should_synthesize_speech: 是否需要合成语音。

        Returns:
            一个包含最终响应的字典。
        """
        response_to_client = {
            "type": intent,
            "should_synthesize_speech": should_synthesize_speech,
            "content": final_user_facing_content
        }
        return {
            "response": response_to_client,
            "full_assistant_response": full_assistant_response
        }

    async def _build_thinking_prompt(self, orchestrator, user_input: str, topic_summary: str, full_session_history: list, tool_result: str = None) -> str:
        """
        根据当前上下文动态构建思考智能体（ThinkingAgent）的System Prompt。
        
        业务逻辑:
        1.  **加载通用信息**:
            -   从AI宪法（Constitution）中加载AI的人设描述。
            -   格式化当前主题的对话历史。
            -   格式化完整的会话历史，包含每次交互的角色、时间、内容和相关主题信息。
        2.  **根据有无工具结果选择模板**:
            -   **有工具结果**: 使用 `with_tool` 模板，将工具名称、参数和结果填入Prompt。
            -   **无工具结果**: 使用 `without_tool` 模板，将AI宪法中定义的输出结构和标签填入Prompt，指导LLM生成格式化的答案。
        3.  **返回最终构建的Prompt字符串**。
        """
        # 加载人设信息
        persona = orchestrator.constitution['persona']
        persona_desc = f"你的名字是 {persona['identity']['name']}，一个 {persona['identity']['role']}。你的核心特质是：{', '.join(persona['identity']['core_traits'])}。"
        
        # 格式化主题历史
        formatted_topic_history_text = "\n".join([f"{item['role']}: {item['content']}" for item in full_session_history[-10:]]) or "无"
        
        # 格式化完整会话历史
        formatted_session_history_lines = []
        for i, turn in enumerate(full_session_history[-20:]):
            turn_str = f"对话 {i+1}：\n角色：{turn.role}\n时间：{turn.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n内容：{turn.content}"
            if turn.role == 'user' and turn.meta_data and 'topic_info' in turn.meta_data:
                topic_info = turn.meta_data['topic_info']
                turn_str += f"\n主题：{topic_info.get('name', '无')}\n主题内容：\n  摘要：{topic_info.get('summary', '无')}"
                for conv_item in topic_info.get('conversations', []):
                    turn_str += f"\n  - {conv_item['role']}: {conv_item['content']}"
            formatted_session_history_lines.append(turn_str)
        
        formatted_session_history_text = "\n\n".join(formatted_session_history_lines) or "无"

        # 准备通用参数
        common_args = {
            "persona_desc": persona_desc, "user_input": user_input,
            "topic_summary": topic_summary or "无", "topic_history_text": formatted_topic_history_text,
            "full_session_history_text": formatted_session_history_text
        }

        # 根据有无工具结果，选择不同的Prompt模板
        if tool_result:
            prompt = orchestrator.prompt_manager.format_prompt(
                agent_name="thinking_agent", template_key="with_tool", tool_result=tool_result, **common_args
            )
        else:
            rules = orchestrator.constitution['input_handling']['type_rules']['提问']
            output_structure = "\n".join([f"- {item}" for item in rules['结构化输出']])
            prompt = orchestrator.prompt_manager.format_prompt(
                agent_name="thinking_agent", template_key="without_tool", output_label=rules['输出标签'],
                output_structure=output_structure, **common_args
            )
        return prompt

    async def _handle_sync_tool_use(self, orchestrator, master_topic: str, decision: Dict[str, Any]) -> str:
        """
        处理同步工具的调用。
        
        业务逻辑:
        1.  从决策字典中提取工具名称和参数。
        2.  在任务处理器（TaskHandler）中创建一个同步任务记录。
        3.  更新任务状态为“运行中”。
        4.  使用`run_in_executor`在独立的线程中执行工具，避免阻塞事件循环。
        5.  执行完毕后，通过WebSocket向前端发送工具的输出结果。
        6.  返回工具执行结果的字符串。
        7.  如果发生异常，记录错误，更新任务状态为“失败”，并返回错误信息。
        """
        tool_name, tool_args = decision.get("tool"), decision.get("args", {})
        task = orchestrator.task_handler.create_task(master_topic, tool_name, tool_args, 'sync')
        task_id = task["id"]
        try:
            orchestrator.task_handler.update_task_status(task_id, 'running')
            loop = asyncio.get_running_loop()
            # 在线程池中执行阻塞的工具调用
            result = await loop.run_in_executor(None, orchestrator.tool_registry.execute_tool, tool_name, tool_args, orchestrator.tool_config)
            orchestrator.task_handler.update_task_status(task_id, 'completed', str(result))
            logger.info(f"向前端发送并保存同步工具执行结果")
            # 通知前端工具执行结果
            await orchestrator.notify_and_save_message(
                session_id=decision.get("session_id"), 
                content=str(result),
                role="assistant_tool_sync_output",
                meta_data={"tool_name": tool_name, "tool_args": tool_args}
            )
            return str(result)
        except Exception as e:
            logger.exception(f"执行同步工具 '{tool_name}' 时发生错误: {e}")
            err_msg = f"错误：使用'{tool_name}'工具时发生内部错误: {str(e)}"
            orchestrator.task_handler.update_task_status(task_id, 'failed', err_msg)
            return err_msg

    async def _run_async_tool_and_notify(self, orchestrator, session_id: str, master_topic: str, task_id: str, tool_name: str, tool_args: Any, session_tools: Dict[str, Any], client_message_id: str | None = None, stream_callback=None):
        """
        在后台执行异步工具，并在完成后通过WebSocket发送通知。
        这是一个被 `_handle_async_tool_use` 创建为后台任务的协程。
        
        业务逻辑:
        1.  更新任务状态为“运行中”。
        2.  在独立的线程中执行工具调用。
        3.  根据工具返回结果，更新任务状态为“完成”或“失败”。
        4.  如果发生异常，记录错误并更新任务状态为“失败”。
        5.  无论成功或失败，最终都会通过WebSocket将结果（或错误信息）发送给前端。
        """
        tool_result, task_status = None, "failed"
        # 预判是否为流式工具，决定完成后是否向前端再次通知完整内容
        is_stream_tool = False
        try:
            tool_info = orchestrator.tool_registry.get_tool_info(tool_name, session_tools)
            if tool_info:
                try:
                    loc = tool_info.location
                    location = json.loads(loc) if isinstance(loc, str) else loc
                    is_stream_tool = bool(location.get("stream_api"))
                except Exception:
                    is_stream_tool = False
        except Exception:
            is_stream_tool = False
        # 若上层提供了统一的 stream_callback，则强制按流式工具处理，避免重复完整通知
        if stream_callback is not None:
            is_stream_tool = True
        try:
            orchestrator.task_handler.update_task_status(task_id, 'running')
            loop = asyncio.get_running_loop()
            # 构建流式桥接函数：线程安全地将片段调度回事件循环并推送到前端
            def bridge_chunk(text: str, phase: str = "processing"):
                try:
                    # 使用事件循环线程安全调度
                    def _schedule():
                        # 若上层提供统一的 stream_callback，则复用统一结构与 message_id
                        if stream_callback:
                            # 将阶段映射到 orchestrator 约定事件类型
                            event_type = {
                                "stream": "content",
                                "processing": "content",
                                "raw": "content",
                                "completed": "end"
                            }.get(phase, "content")
                            return asyncio.create_task(stream_callback(text or "", event_type))
                        # 否则回退到直接通知（保持此前的非重复保存策略）
                        if phase == "completed":
                            return
                        msg = {
                            "type": "newChatMessage",
                            "data": {
                                "session_id": session_id,
                                "message_id": client_message_id or f"{task_id}-stream",
                                "role": "assistant_tool_stream_output",
                                "timestamp": int(time.time()),
                                "content": text,
                                "type": "text",
                                "tool_calls": None,
                                "meta_data": {"task_id": task_id, "phase": phase, "tool_name": tool_name}
                            }
                        }
                        asyncio.create_task(orchestrator.notify_client(session_id, msg))
                    loop.call_soon_threadsafe(_schedule)
                except Exception as e:
                    logger.warning(f"桥接流式片段失败: {e}")

            # 在线程池中执行可能阻塞的工具调用，扩展SSE路径以传入桥接回调
            def _exec():
                try:
                    # 访问底层执行以传递 on_chunk 回调（仅当为外部API且为SSE）
                    # 通过 registry.get_tool_info 判断是否为stream_api
                    t = orchestrator.tool_registry.get_tool_info(tool_name, session_tools) or orchestrator.tool_registry.get_tool_info(tool_name)
                    if t:
                        loc = json.loads(t.location) if isinstance(t.location, str) else t.location
                        if isinstance(loc, dict) and loc.get("stream_api"):
                            return orchestrator.tool_registry._execute_sse_api(t, tool_args, loc, on_chunk=bridge_chunk)
                    # 回退到统一入口
                    return orchestrator.tool_registry.execute_tool(tool_name, tool_args, session_tools)
                except Exception as e:
                    logger.exception(f"工具执行异常: {e}")
                    raise

            tool_result = await loop.run_in_executor(None, _exec)
            task_status = "failed" if isinstance(tool_result, str) and tool_result.startswith("错误：") else "completed"
            orchestrator.task_handler.update_task_status(task_id, task_status, tool_result)
        except Exception as e:
            logger.exception(f"处理异步任务{task_id}时发生严重错误: {e}")
            tool_result = f"错误：在后台处理'{tool_name}'工具时发生了内部错误: {str(e)}"
            orchestrator.task_handler.update_task_status(task_id, 'failed', tool_result)
        finally:
            logger.info(f"异步任务 {task_id} 完成，准备通知前端。结果: {tool_result}")
            # 对于流式工具：已通过 on_chunk 将片段逐步推送给前端，为避免重复展示，完成后只保存，不再次通知
            if is_stream_tool:
                try:
                    db_message = {
                        "session_id": session_id,
                        "message_id": client_message_id or str(uuid.uuid4()),
                        "role": "assistant",
                        "content": str(tool_result),
                        "timestamp": int(time.time()),
                        "type": "md",
                        "tool_calls": None,
                        "meta_data": {"task_id": task_id, "tool_name": tool_name, "tool_args": tool_args, "source": "stream"}
                    }
                    await asyncio.to_thread(chat_history_repository.add_message_to_history, db_message)
                except Exception as e:
                    logger.error(f"保存流式工具完整结果到历史失败: {e}", exc_info=True)
            else:
                # 非流式工具：仍按旧逻辑通知并保存完整结果
                await orchestrator.notify_and_save_message(
                    session_id=session_id,
                    content=tool_result,
                    role="assistant",
                    meta_data={"task_id": task_id, "tool_name": tool_name, "tool_args": tool_args}
                )
            # 无论是否为流式工具，只要存在统一的回调，则在完成时发送结束事件，避免遗漏
            try:
                if stream_callback:
                    await stream_callback("", "end")
            except Exception:
                pass

    async def _handle_async_tool_use(self, orchestrator, session_id: str, master_topic: str, decision: Dict[str, Any]) -> str:
        """
        处理异步工具的调用。
        
        业务逻辑:
        1.  从决策字典中提取工具名称和参数。
        2.  根据工具需求，为特定工具扩充参数。
        3.  在任务处理器中创建一个异步任务记录。
        4.  使用 `asyncio.create_task` 创建一个新的后台任务，该任务将执行 `_run_async_tool_and_notify` 协程。
        5.  **立即**返回一个友好的提示信息给用户，告知任务已在后台开始。
        """
        tool_name, tool_args = decision.get("tool"), decision.get("args", {})

        # [更新逻辑] 为特定的SSE工具强制覆盖参数，确保使用正确的值
        if tool_name == "japan_aquaculture_expert":
            tool_args["session_id"] = session_id
            # 若缺失 query/agent_type，则补充，避免服务端等待历史检索导致首事件超时
            tool_args.setdefault("query", decision.get("args", {}).get("query"))
            tool_args.setdefault("agent_type", decision.get("args", {}).get("agent_type", "japan"))
            # 透传客户端提供的 message_id（如果存在）
            if "message_id" in decision.get("args", {}):
                tool_args["message_id"] = decision["args"]["message_id"]
            if not tool_args["query"]:
                # 回退策略：从会话最近一条用户消息取问题
                try:
                    recent_user_msg = chat_history_repository.get_latest_user_message(session_id)
                    tool_args["query"] = recent_user_msg or "日本陆上养殖"
                except Exception:
                    tool_args["query"] = "日本陆上养殖"
            # 使用工具支持的模式：single 或 auto。
            # 此处默认选择 single，以避免“Invalid mode: roleplay”错误。
            tool_args["config"] = {
                "rag": {
                    "collection_name": "japan_shrimp",
                    "topk_single": 5,
                    "topk_multi": 5
                },
                "mode": "single",
                "single": {
                    "temperature": 0.4,
                    "system_prompt": "你是一个领域专家，你的任务是根据用户的问题，结合增强检索后的相关知识，给出专业的回答。",
                    "max_tokens": 4096
                }
            }

            logger.info(f"为SSE工具 '{tool_name}' 强制更新参数: {tool_args}")

        task = orchestrator.task_handler.create_task(master_topic, tool_name, tool_args, 'async')
        # 创建后台任务来执行工具并发送通知，不会阻塞当前流程
        asyncio.create_task(self._run_async_tool_and_notify(
            orchestrator, session_id, master_topic, task.task_id, tool_name, tool_args, orchestrator.tool_config,
            client_message_id=tool_args.get("message_id")
        ))
        # 立即返回给用户的信息
        return f"好的，我已经开始在后台为您执行“{tool_name}”任务。这可能需要一些时间，完成后我在这里通知您。"

    async def _handle_async_tool_use_stream(self, orchestrator, session_id: str, master_topic: str, decision: Dict[str, Any], stream_callback=None) -> str:
        """
        处理异步工具的流式调用。

        设计目标：
        - 将流式工具的片段通过统一的 `stream_callback` 推送给前端，确保在同一 `message_id` 下增量展示。
        - 在后台任务完成后，仅保存完整内容到历史，不再发送冗余的完整消息。

        输入参数：
        - orchestrator: 认知协调器实例
        - session_id: 会话ID
        - master_topic: 主题名称
        - decision: 路由决策字典（包含 tool、args、mode 等）
        - stream_callback: 统一的流式回调（由 orchestrator 提供），形如 callback(text: str, event_type: str)

        输出结果：
        - str: 立即返回的提示语，告知任务已在后台开始执行
        """
        tool_name, tool_args = decision.get("tool"), decision.get("args", {})

        # 为特定的SSE工具强制覆盖参数，确保使用正确的值
        if tool_name == "japan_aquaculture_expert":
            tool_args["session_id"] = session_id
            tool_args.setdefault("query", decision.get("args", {}).get("query"))
            tool_args.setdefault("agent_type", decision.get("args", {}).get("agent_type", "japan"))
            if "message_id" in decision.get("args", {}):
                tool_args["message_id"] = decision["args"]["message_id"]
            if not tool_args["query"]:
                try:
                    recent_user_msg = chat_history_repository.get_latest_user_message(session_id)
                    tool_args["query"] = recent_user_msg or "日本陆上养殖"
                except Exception:
                    tool_args["query"] = "日本陆上养殖"
            tool_args["config"] = {
                "rag": {
                    "collection_name": "japan_shrimp",
                    "topk_single": 5,
                    "topk_multi": 5
                },
                "mode": "single",
                "single": {
                    "temperature": 0.4,
                    "system_prompt": "你是一个领域专家，你的任务是根据用户的问题，结合增强检索后的相关知识，给出专业的回答。",
                    "max_tokens": 4096
                }
            }
            logger.info(f"为SSE工具 '{tool_name}' 注入流式参数: {tool_args}")

        task = orchestrator.task_handler.create_task(master_topic, tool_name, tool_args, 'async')
        # 创建后台任务，传入流式回调；片段将通过统一的 stream_callback 推送
        asyncio.create_task(self._run_async_tool_and_notify(
            orchestrator, session_id, master_topic, task.task_id, tool_name, tool_args, orchestrator.tool_config,
            client_message_id=tool_args.get("message_id"), stream_callback=stream_callback
        ))
        return f"好的，我已经开始在后台为您执行“{tool_name}”任务。这可能需要一些时间，完成后我在这里通知您。"
