# cognitive_model/orchestrator.py (v4.0 - Refactored)

import logging, json, os, datetime
from typing import Dict, Any, Callable
import uuid
import time
import asyncio

# 导入所有Agents
from .agents.intent_agent import IntentAgent
from .agents.routing_agent import RoutingAgent
from .agents.classification_agent import ClassificationAgent
from .agents.thinking_agent import ThinkingAgent
from .agents.summarization_agent import SummarizationAgent
from .agents.cognitive_tuner_agent import CognitiveTunerAgent

# 导入核心模块
from .hippocampus.handler import HippocampusHandler
from .hippocampus import session_handler
from .tools.tool_registry import ToolRegistry
from .tasks.task_handler import TaskHandler
from .config.prompt_manager import PromptManager

# 导入新的Handlers和状态管理器
from .handlers.session_state_manager import SessionStateManager
from .handlers.query_handler import QueryHandler
from .handlers.tuning_handler import TuningHandler
from .handlers.simple_handler import SimpleHandler

# 导入数据库仓库层和会话服务
from singa_one_server.repositories import chat_history_repository, session_repository
from singa_one_server.services import session_service
from singa_one_server.utils.websocket_utils import chat_std
from singa_one_server.utils.time_domain import get_chinese_date_string, get_chinese_datetime_string, get_chinese_datetime_string_jst, calc_prepend_timestamp

logger = logging.getLogger("app.cognitive_model.orchestrator")

class CognitiveOrchestrator:
    """
        认知协调器 (CognitiveOrchestrator) v4.2 - 动态工具加载

        作为认知模型的核心中枢，该类负责编排整个认知流程。它接收用户输入，协调多个专门化的智能体（Agents）
        和核心功能模块（如海马体记忆、任务处理器、工具注册表等），以理解用户意图、处理任务并生成最终响应。

        核心设计思想:
        1.  **策略模式 (Strategy Pattern)**: 采用策略模式，根据 `IntentAgent` 识别出的用户意图，将控制权
            动态地分发给相应的处理器（Handler）。这种设计使得系统易于扩展，新增或修改意图处理逻辑时，
            只需实现新的 Handler 或修改现有 Handler，而无需改动 Orchestrator 的核心代码。
        2.  **状态驱动 (State-Driven)**: 引入 `SessionStateManager` 来管理会话状态。这使得系统能够处理
            多轮对话中的复杂交互，例如，当一个需要用户确认的计划（如代码修改）被创建后，系统会进入
            一个"待定"状态，优先处理用户的确认，而不是重新识别意图。
        3.  **模块化与责任分离**: 将不同的认知功能（如意图识别、路由、分类、思考、总结、微调）封装在
            各自的 Agent 中。将不同的业务处理逻辑（如普通查询、系统微调、简单交互）封装在 Handler 中。
            这种清晰的责任划分提高了代码的可维护性和可测试性。
        4.  **统一通信与持久化**: 提供了 `notify_and_save_message` 方法，作为向客户端发送实时通知和
            向数据库保存消息的统一出口。这确保了所有与用户的交互（包括中间步骤）都被可靠地记录下来。
        5.  **动态工具加载**: 采用分级工具加载机制，系统启动时只加载默认工具，在处理每个请求时根据会话配置
            动态加载自定义工具。这种设计提高了系统的灵活性和安全性，允许不同会话使用不同的工具集。
    """
    def __init__(self, notification_callback: Callable):
        """
            初始化认知协调器。

            此构造函数负责实例化所有必要的组件，包括各种智能体、核心模块、处理器，并加载基础配置。

            业务逻辑:
            1.  **组件初始化**: 创建所有 Agent 和核心模块的实例。这些组件是执行认知任务的基础。
            2.  **加载AI宪法**: 从数据库加载 "AI宪法"。这个 "AI宪法" 包含了
                指导模型行为的核心原则和高级指令（如角色、语气、安全约束等），是生成所有 Prompt 的基础。
            3.  **初始化处理器**: 创建状态管理器 (`SessionStateManager`) 和各种意图处理器
                (`QueryHandler`, `TuningHandler`, `SimpleHandler`) 的实例。
            4.  **建立意图映射**: 创建 `handler_map` 字典，将识别出的意图字符串（如 "提问"）映射到
                对应的处理器对象。这是策略模式实现的核心。
            5.  **初始化统计**: 设置用于追踪 Token 消耗的数据结构。

            Args:
                notification_callback (Callable): 用于向客户端发送实时通知的回调函数。
        """
        self.prompt_manager = PromptManager()
        self.tool_registry = ToolRegistry()  
        
        self.intent_agent = IntentAgent(prompt_manager=self.prompt_manager)
        self.routing_agent = RoutingAgent(prompt_manager=self.prompt_manager, tool_registry=self.tool_registry)
        self.classification_agent = ClassificationAgent(prompt_manager=self.prompt_manager)
        self.thinking_agent = ThinkingAgent(prompt_manager=self.prompt_manager)
        self.summarization_agent = SummarizationAgent(prompt_manager=self.prompt_manager)
        self.cognitive_tuner_agent = CognitiveTunerAgent(prompt_manager=self.prompt_manager)

        self.session_state_manager = SessionStateManager()
        self.query_handler = QueryHandler()
        self.tuning_handler = TuningHandler(prompt_manager=self.prompt_manager)
        self.simple_handler = SimpleHandler()
        
        self.task_handler = TaskHandler()
        self.hippocampus = HippocampusHandler(prompt_manager=self.prompt_manager)
        
        self.handler_map = {
            "提问": self.query_handler,
            "反馈": self.query_handler,
            "调整": self.tuning_handler,
            "自我介绍": self.simple_handler,
            "唤醒": self.simple_handler,
            "通知": self.simple_handler,
        }

        self.notify_client = notification_callback
        self.total_token_stats = {}
        self.model_config = {}
        self.tool_config = []
        self.rag_config = []
        self.token_count = 0
        self.summary_amount = 5000
        self.handler_result = None
        self.history = []
        self.history_limit = None
        self.constitution = None

    def _update_total_stats(self, agent_name: str, stats: Dict[str, Any]):
        """
        统一更新和聚合来自不同Agent的Token统计数据。

        这是一个辅助方法，用于在每次 Agent 调用后，累积其 Token 使用量。
        这对于成本控制、性能监控和分析模型的使用情况至关重要。

        Args:
            agent_name (str): 产生消耗的Agent名称 (e.g., "intent_agent")。
            stats (Dict[str, Any]): Agent返回的统计数据，必须包含 'total_tokens' 键。
        """
        if agent_name not in self.total_token_stats:
            self.total_token_stats[agent_name] = {"total_calls": 0, "total_tokens": 0}
        self.total_token_stats[agent_name]["total_calls"] += 1
        self.total_token_stats[agent_name]["total_tokens"] += stats.get("total_tokens", 0)

    def _update_tool_config(self, custom_tool_config: list) -> Dict[str, Any]:
        custom_tools = self.tool_registry.get_tools_by_ids(custom_tool_config)
        # 默认工具与私人工具合并
        final_tool_config = {**self.tool_registry._tools_by_id, **custom_tools}  
        return final_tool_config
    
    async def notify_and_save_message(
        self, 
        session_id: str, 
        content: Any, 
        role: str = "tool", 
        msg_type: str = "text", 
        tool_calls: list = None, 
        meta_data: dict = None
        ):
        """
            统一的消息发送和持久化方法。
            
            Args:
                session_id (str): 会话ID。
                content (Any): 消息内容。
                role (str, optional): 角色。默认为 "assistant_tool_output"。
                msg_type (str, optional): 消息类型。默认为 "text"。
                tool_calls (list, optional): 工具调用列表。默认为 None。
                meta_data (dict, optional): 附加的元数据。
        """
        message_id = str(uuid.uuid4())
        
        # 1. 准备要发送给客户端的消息
        notification_message = {
            "type": "newChatMessage", 
            "data": {
                "session_id": session_id,
                "message_id": message_id,
                "role": role, 
                "timestamp": int(time.time()),
                "content": content,
                "type": msg_type,
                "tool_calls": tool_calls,
                "meta_data": meta_data
            }
        }
        
        # 2. 准备要保存到数据库的回合数据
        db_message = {
            "session_id": session_id,
            "message_id": message_id,
            "role": role,
            "content": str(content), 
            "timestamp": int(time.time()),
            "type": msg_type,
            "tool_calls": tool_calls,
            "meta_data": meta_data
        }
        
        # 3. 并发执行发送和保存操作，提高响应速度
        try:
            await asyncio.gather(
                self.notify_client(session_id, notification_message),
                asyncio.to_thread(chat_history_repository.add_message_to_history, db_message)
            )
            # logger.info(f"消息已成功发送并保存至会话 {session_id}。")
        except Exception as e:
            logger.error(f"为会话 {session_id} 发送并保存消息时出错: {e}", exc_info=True)
            
    async def _get_or_create_session(self, session_id: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        获取或创建会话。

        这是一个关键的辅助方法，用于在处理任何用户输入之前，确保会话在数据库中存在。
        它封装了与 `session_service` 的交互，实现了会话的按需创建和加载。

        业务逻辑:
        1.  调用 `session_service.get_or_create_session`，传入 `session_id` 和 `user_id`。
        2.  服务层会处理数据库查找或创建的逻辑。
        3.  返回从数据库中获取或新创建的会话对象。

        Args:
            session_id (str): 会话的唯一标识符。
            user_id (str): 用户的唯一标识符，默认为 "default_user"。

        Returns:
            Dict[str, Any]: 代表会话的字典对象。
        """
        # logger.warning(f"正在为 session_id: {session_id} 获取或创建会话...")
        loop = asyncio.get_running_loop()
        session = await loop.run_in_executor(
            None, session_service.initialize_session, session_id, user_id
        )
        # logger.warning(f"会话 {session_id} 已成功加载或创建。")
        return session

    async def process_input(self, user_input: str, websocket_id: str, session_id: str) -> None:
        """
            处理用户输入的核心入口点和总编排方法。

            这是 Orchestrator 的核心，它定义了处理用户请求的完整生命周期。

            业务逻辑 (按顺序执行):
            1.  **会话管理**:
                -   调用 `_get_or_create_session` 确保会话存在于数据库中。
                -   将用户的原始输入立即存入数据库。

            2.  **初始化与重置**:
                -   重置本轮对话的 Token 统计数据。
                -   根据会话配置中的 `tool_config` 动态加载工具。

            3.  **状态优先检查 (State-First Check)**:
                -   检查 `SessionStateManager` 中是否存在待处理的计划 (`pending_update_plan`)。
                -   如果存在，强制将任务路由到 `TuningHandler` 处理用户确认。

            4.  **意图驱动处理 (Intent-Driven Process)**:
                -   如果不存在待处理的状态，则调用 `IntentAgent` 识别用户意图。
                -   使用 `handler_map` 查找到与意图对应的处理器 (Handler)。
                -   调用该 Handler 的 `handle` 方法，将控制权完全交给它。

            5.  **统一后处理 (Unified Post-Processing)**:
                -   从 Handler 的返回结果中提取最终响应和完整执行过程。
                -   将AI完整的、未经删减的最终回应 (`full_assistant_response`) 保存到数据库。

            Args:
                user_input (str): 用户输入的原始文本。
                websocket_id (str): 当前连接的WebSocket ID。
                session_id (str): 当前的会话ID。
        """
        logger.warning(f"[会话 {session_id}] --- 开始新一轮认知处理 (v4.2 动态工具加载) ---")
                
        self.constitution = self.prompt_manager.get_prompt("constitution","system")   
        logger.info(f"'宪法'加载成功!")
        
        user_message = chat_std(session_id, user_input, "text", "user")
        await asyncio.to_thread(chat_history_repository.add_message_to_history, user_message)
        
        session = await self._get_or_create_session(session_id)
        config = session.get("config", {})
        
        self.model_config = config.get("model", {})
        self.rag_config = config.get("rag", [])
        self.token_count = config.get("token_count", 0)
        self.summary_amount = config.get("summary_amount", 5000)
        self.tool_config = self._update_tool_config(config.get("tool", []) )
        self.history = await asyncio.to_thread(chat_history_repository.get_history_by_session_id, session_id, limit=self.history_limit)

        # 在意图识别前进行历史预处理：确保存在“在日本陆上养殖中”提示
        await self._ensure_japan_land_based_phrase_in_history(session_id)
        # 预处理后刷新历史以便意图识别使用最新顺序
        self.history = await asyncio.to_thread(chat_history_repository.get_history_by_session_id, session_id, limit=self.history_limit)
        # 继续进行时间域预处理：确保存在当天日期提示
        await self._ensure_date_phrase_in_history(session_id)
        # 再次刷新历史，确保最新插入位于最开头
        self.history = await asyncio.to_thread(chat_history_repository.get_history_by_session_id, session_id, limit=self.history_limit)
        
        # 构造增强后的用户输入（在最开始加入两条短语）
        japan_phrase = "在日本陆上养殖中"
        # 使用JST（日本标准时间）生成时间短语，保证前端显示与业务场景一致
        date_phrase = f"现在是{get_chinese_datetime_string_jst()}"
        augmented_input = f"{japan_phrase} {date_phrase} {user_input}"
        
        # logger.warning(f"会话 {session_id} 加载配置: 模型={self.model_config}, 工具={self.tool_config}, RAG={self.rag_config}, Token数={self.token_count}, 摘要量={self.summary_amount}")

        pending_plan = self.session_state_manager.get_state(session_id, 'pending_update_plan')
        if pending_plan:
            logger.info(f"发现会话 {session_id} 中有待处理的计划，将强制路由到TuningHandler。")
            handler = self.tuning_handler
            self.handler_result = await handler.handle(
                self, user_input, session_id
            )
        else:
            intent, intent_stats = await self.intent_agent.get_intent(augmented_input, self.history, model_config=self.model_config)
            self._update_total_stats("intent_agent", intent_stats)

            handler = self.handler_map.get(intent)
            if handler:
                self.handler_result = await handler.handle(
                    self, augmented_input, session_id, session=session, intent=intent
                )
            else:
                logger.error(f"未找到意图 '{intent}' 对应的处理器。")
                response_to_client = {"type": "错误", "should_synthesize_speech": True, "content": "抱歉，我无法理解您的意图。"}
                self.handler_result = {"response": response_to_client, "full_assistant_response": "Error: No handler for intent."}

        # 保存完整的AI助手回应
        if self.handler_result:
            await self.notify_and_save_message(
                session_id=session_id,
                content=self.handler_result.get("full_assistant_response"),
                role="assistant",
                meta_data=self.handler_result.get("meta_data")
            )
        logger.warning(f"[本轮会话 {session_id}] --- 认知处理结束 ---")
        return

    async def _ensure_japan_land_based_phrase_in_history(self, session_id: str) -> None:
        """
        会话历史预处理：确保在意图识别前，历史记录中包含“在日本陆上养殖中”。

        业务逻辑:
        1. 检查当前加载的历史（self.history）中是否已出现该短语（任意角色）。
        2. 若不存在，则在历史的开头插入一条角色为“user”的消息，内容为“在日本陆上养殖中”。
           为确保位于“开头”，该插入消息的时间戳设置为最早一条消息时间戳的前一秒；若历史为空，则使用 0（1970-01-01）。

        输入:
        - session_id (str): 会话ID，用于持久化插入的历史消息。

        输出:
        - None。该方法通过数据库仓库写入消息，并不直接返回数据。

        关键实现细节:
        - 历史对象类型为 ORM（ChatHistory），通过 getattr 访问 content；
        - 使用工具函数 chat_std 构造标准消息后覆盖 timestamp，确保“开头”顺序；
        - 插入后不在此方法中刷新 self.history，由调用方在调用后统一刷新。
        """
        try:
            # 1) 检查是否已存在该短语
            target_phrase = "在日本陆上养殖中"
            exists = False
            for msg in self.history or []:
                # ChatHistory.timestamp 为 datetime；content 为字符串
                content = getattr(msg, 'content', '')
                if isinstance(content, str) and target_phrase in content:
                    exists = True
                    break

            if exists:
                return

            # 2) 计算插入时间戳（最早一条的前一秒；若无历史则置 0）
            if self.history:
                earliest_dt = getattr(self.history[0], 'timestamp', None)
                if earliest_dt is not None:
                    earliest_ts = int(earliest_dt.timestamp())
                    insert_ts = max(earliest_ts - 1, 0)
                else:
                    insert_ts = 0
            else:
                insert_ts = 0

            # 3) 构造并写入“开头”消息（role=user）
            injected = chat_std(session_id, target_phrase, "text", "user")
            injected['timestamp'] = insert_ts
            injected['meta_data'] = {"injected": True, "reason": "force_japan_tool"}
            await asyncio.to_thread(chat_history_repository.add_message_to_history, injected)
        except Exception as e:
            logger.error(f"会话 {session_id} 历史预处理失败: {e}", exc_info=True)

    async def _ensure_date_phrase_in_history(self, session_id: str) -> None:
        """
        会话历史预处理：确保在意图识别前，历史记录中包含当天日期的中文提示。

        业务逻辑:
        1. 生成当前日期中文字符串，格式为 "YYYY年MM月DD日HH时MM分"，不含“现在是”前缀。
        2. 检查当前历史（self.history）是否已包含当天日期字符串（任意角色）。
        3. 若不存在，则在历史开头插入一条角色为 "user" 的日期消息。
           插入时间戳通过 `calc_prepend_timestamp(self.history)` 计算，确保位于最开头。

        输入:
        - session_id (str): 会话ID，用于持久化插入的历史消息。

        输出:
        - None。该方法通过数据库仓库写入消息，并不直接返回数据。

        关键实现细节:
        - 使用 `get_chinese_date_string()` 生成日期字符串（UTC 基准）。
        - 仅在当天日期不存在时插入；若历史已有其他日期（如昨日），新增插入视为“更新”时间域。
        - 插入后不在此方法中刷新 self.history，由调用方在调用后统一刷新。
        """
        try:
            # 使用日期时间（到分钟）作为时间域提示，加入“现在是”前缀
            # 插入历史的日期短语使用JST版本，保持显示一致；数据库仍以UTC时间戳保存
            date_phrase = f"现在是{get_chinese_datetime_string_jst()}"

            # 检查当天日期是否已存在
            exists_today = False
            for msg in self.history or []:
                content = getattr(msg, 'content', '')
                if isinstance(content, str) and date_phrase in content:
                    exists_today = True
                    break

            if exists_today:
                return

            # 计算插入时间戳，保证最开头
            insert_ts = calc_prepend_timestamp(self.history)

            injected = chat_std(session_id, date_phrase, "text", "user")
            injected['timestamp'] = insert_ts
            injected['meta_data'] = {"injected": True, "reason": "time_domain_date"}
            await asyncio.to_thread(chat_history_repository.add_message_to_history, injected)
        except Exception as e:
            logger.error(f"会话 {session_id} 时间域预处理失败: {e}", exc_info=True)

    async def process_input_stream(self, user_input: str, websocket_id: str, session_id: str, client_message_id: str | None = None):
        """
            以流式方式处理用户输入，并实时返回响应。

            此方法是为需要实时反馈的场景设计的，例如聊天界面。它通过回调函数将LLM生成的文本块
            （chunk）陆续发送给客户端，而不是等待完整的响应生成后再发送。

            业务逻辑:
            1.  **初始化与设置**: 与 `process_input` 类似，加载会话、配置和历史记录。
            2.  **定义流式回调**: 创建一个嵌套的异步函数 `stream_callback`。此函数负责：
                - 接收来自Agent的流式数据块 (`chunk`)。
                - 将数据块累加到 `full_assistant_response` 以便最后完整保存。
                - 构建一个标准化的消息体 (`stream_chunk`)。
                - 调用 `self.notify_client` 将消息实时发送给客户端。
            3.  **意图识别与路由**:
                - 识别用户意图。
                - **如果意图是 "提问"**，则调用 `thinking_agent.run_stream`，并将 `stream_callback`
                作为参数传入。这是实现流式响应的核心。
                - **对于其他意图**，系统会回退到非流式处理，并告知用户当前功能不支持流式响应。
            4.  **错误处理**: 捕获处理过程中的任何异常，并通过 `notify_client` 向客户端发送错误信息。
            5.  **最终持久化**:
                - 在流式处理结束后（无论成功或失败），将拼接完成的 `full_assistant_response`
                完整地保存到数据库中。这确保了即使是流式对话，历史记录也是完整的。

            Args:
                user_input (str): 用户输入的原始文本。
                websocket_id (str): 当前连接的WebSocket ID。
                session_id (str): 当前的会话ID。
        """
        logger.warning(f"[会话 {session_id}] --- 开始新一轮流式认知处理 ---")

        self.constitution = self.prompt_manager.get_prompt("constitution", "system")
        
        user_message = chat_std(session_id, user_input, "text", "user")
        await asyncio.to_thread(chat_history_repository.add_message_to_history, user_message)

        session = await self._get_or_create_session(session_id)
        config = session.get("config", {})

        self.model_config = config.get("model", {})
        self.tool_config = self._update_tool_config(config.get("tool", []))
        self.history = await asyncio.to_thread(chat_history_repository.get_history_by_session_id, session_id, limit=self.history_limit)

        # 在意图识别前进行历史预处理：确保存在“在日本陆上养殖中”提示
        await self._ensure_japan_land_based_phrase_in_history(session_id)
        # 预处理后刷新历史以便意图识别使用最新顺序
        self.history = await asyncio.to_thread(chat_history_repository.get_history_by_session_id, session_id, limit=self.history_limit)
        # 继续进行时间域预处理：确保存在当天日期提示
        await self._ensure_date_phrase_in_history(session_id)
        # 再次刷新历史，确保最新插入位于最开头
        self.history = await asyncio.to_thread(chat_history_repository.get_history_by_session_id, session_id, limit=self.history_limit)

        # 构造增强后的用户输入（在最开始加入两条短语）
        japan_phrase = "在日本陆上养殖中"
        # 流式路径也采用JST时间短语
        date_phrase = f"现在是{get_chinese_datetime_string_jst()}"
        augmented_input = f"{japan_phrase} {date_phrase} {user_input}"

        # 若客户端提供 message_id，则复用该ID以实现端到端一致性；否则生成新的内部ID
        message_id = client_message_id or str(uuid.uuid4())
        full_assistant_response = ""

        async def stream_callback(chunk: str, event_type: str):
            nonlocal full_assistant_response
            if chunk:
                full_assistant_response += chunk
            
            message = {
                "type": "stream_chunk",
                "data": {
                    "session_id": session_id,
                    "message_id": message_id,
                    "role": "assistant",
                    "timestamp": int(time.time()),
                    "content": chunk,
                    "event": event_type, # 'content', 'tool_call', or 'end'
                }
            }
            await self.notify_client(session_id, message)

        try:
            intent, intent_stats = await self.intent_agent.get_intent(augmented_input, self.history, model_config=self.model_config)
            self._update_total_stats("intent_agent", intent_stats)

            if intent == "提问":
                logger.info(f"意图为 '提问'，启动流式处理。")
                # 使用QueryHandler的流式版本
                handler_result = await self.query_handler.handle_stream(
                    orchestrator=self,
                    user_input=augmented_input,
                    session_id=session_id,
                    stream_callback=stream_callback,
                    intent=intent,
                    model_config=self.model_config,
                    message_id=message_id
                )
                full_assistant_response = handler_result.get("full_assistant_response", "")
            elif intent in ["反馈", "自我介绍", "唤醒", "通知"]:
                logger.info(f"意图为 '{intent}'，启动流式处理。")
                # 使用SimpleHandler的流式版本
                handler_result = await self.simple_handler.handle_stream(
                    orchestrator=self,
                    user_input=user_input,
                    session_id=session_id,
                    stream_callback=stream_callback,
                    intent=intent,
                    model_config=self.model_config
                )
                full_assistant_response = handler_result.get("full_assistant_response", "")
            elif intent == "调整":
                logger.info(f"意图为 '调整'，启动流式处理。")
                # 使用TuningHandler的流式版本
                handler_result = await self.tuning_handler.handle_stream(
                    orchestrator=self,
                    user_input=user_input,
                    session_id=session_id,
                    stream_callback=stream_callback,
                    intent=intent,
                    model_config=self.model_config
                )
                full_assistant_response = handler_result.get("full_assistant_response", "")
            else:
                logger.info(f"意图 '{intent}' 不支持流式响应，回退到标准处理。")
                response_content = f"抱歉，'{intent}'功能暂不支持流式响应。"
                full_assistant_response = response_content
                await stream_callback(response_content, "content")

        except Exception as e:
            logger.error(f"处理流式输入时出错: {e}", exc_info=True)
            error_message = {"type": "error", "data": {"content": "处理您的请求时发生错误。"}}
            await self.notify_client(session_id, error_message)
            full_assistant_response = f"Error: {e}"

        finally:
            # 确保即使发生错误，也能尝试保存已收到的部分
            if full_assistant_response:
                db_message = {
                    "session_id": session_id,
                    "message_id": message_id,
                    "role": "assistant",
                    "content": str(full_assistant_response),
                    "timestamp": int(time.time()),
                    "type": "md",
                    "tool_calls": None, # TODO: 从流中解析工具调用
                    "meta_data": {"source": "stream", "tokens": len(full_assistant_response)} # 简单的token计数
                }
                await asyncio.to_thread(chat_history_repository.add_message_to_history, db_message)

            logger.warning(f"[本轮流式会话 {session_id}] --- 认知处理结束 ---")
