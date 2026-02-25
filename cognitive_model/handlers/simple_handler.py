# cognitive_model/handlers/simple_handler.py

import logging
from .base_handler import BaseHandler
from typing import Dict, Any, Optional, List

logger = logging.getLogger("app.handler.simple")

class SimpleHandler(BaseHandler):
    """
    处理如"自我介绍"、"唤醒"、"通知"等逻辑相对简单的意图。
    这些意图通常不需要复杂的路由决策或工具调用，可以直接生成回复或执行简单操作。
    """
    
    async def handle_stream(self, orchestrator, user_input: str, session_id: str, stream_callback=None, **kwargs) -> Dict[str, Any]:
        """
        处理简单意图的流式回复版本。
        
        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入。
            session_id: 当前的会话ID。
            stream_callback: 流式回调函数，用于实时发送响应片段。
            **kwargs: 包含 `intent` 的额外参数。
            
        Returns:
            一个字典，包含发送给客户端的响应和完整的AI回复。
        """
        intent = kwargs.get("intent")
        logger.info(f"会话 {session_id}: 使用SimpleHandler流式处理意图 '{intent}'")

        if intent == "自我介绍":
            return await self._handle_self_introduction_stream(orchestrator, user_input, session_id, stream_callback, **kwargs)
        
        elif intent == "唤醒":
            content = "你好，我是one，有什么可以帮助你的吗？"
            # 对于固定回复，也进行流式发送
            if stream_callback:
                await stream_callback(content, "content")
            response = {"type": "唤醒", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}
            
        elif intent == "通知":
            content = "信息已记录。"
            # 对于固定回复，也进行流式发送
            if stream_callback:
                await stream_callback(content, "content")
            response = {"type": "通知", "should_synthesize_speech": False, "content": content}
            return {"response": response, "full_assistant_response": content}
            
        else:
            # 作为备用，处理未知的简单意图
            content = "抱歉，我暂时无法处理这个请求。"
            logger.warning(f"SimpleHandler接收到未明确处理的意图: {intent}")
            if stream_callback:
                await stream_callback(content, "content")
            response = {"type": "错误", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}
    
    async def handle(self, orchestrator, user_input: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        根据传入的意图，分发到不同的处理方法。

        业务逻辑:
        1.  从 `kwargs` 中获取由 `IntentAgent` 识别出的意图（intent）。
        2.  根据意图的值，选择相应的处理分支：
            -   **自我介绍**: 调用 `_handle_self_introduction` 方法，该方法会构建一个特定的Prompt，让LLM生成一段自我介绍，并更新记忆。
            -   **唤醒**: 返回一个预设的、固定的欢迎语。
            -   **通知**: 返回一个预设的、固定的确认信息。
        3.  如果意图不在预设分支中，则返回一个默认的错误或备用响应。

        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入。
            session_id: 当前的会话ID。
            **kwargs: 包含 `intent` 的额外参数。

        Returns:
            一个字典，包含发送给客户端的响应和完整的AI回复。
        """
        intent = kwargs.get("intent")
        logger.info(f"会话 {session_id}: 使用SimpleHandler处理意图 '{intent}'")

        if intent == "自我介绍":
            return await self._handle_self_introduction(orchestrator, user_input, session_id, **kwargs)
        
        elif intent == "唤醒":
            content = "你好，我是one，有什么可以帮助你的吗？"
            response = {"type": "唤醒", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}
            
        elif intent == "通知":
            content = "信息已记录。"
            response = {"type": "通知", "should_synthesize_speech": False, "content": content}
            return {"response": response, "full_assistant_response": content}
            
        else:
            # 作为备用，处理未知的简单意图
            content = "抱歉，我暂时无法处理这个请求。"
            logger.warning(f"SimpleHandler接收到未明确处理的意图: {intent}")
            response = {"type": "错误", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}

    def _build_self_intro_prompt(self, orchestrator, user_input: str) -> str:
        """
        为ThinkingAgent构建用于生成自我介绍的System Prompt。

        业务逻辑:
        1.  从AI宪法（Constitution）中提取AI的人设信息，包括名字、角色、核心特质和使命。
        2.  将这些信息与用户的具体问题组合成一个结构化的Prompt。
        3.  该Prompt明确指示LLM基于其设定的角色和任务，生成一段简洁、友好且富有洞察力的自我介绍。

        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始提问。

        Returns:
            一个完整的、可以直接用作System Prompt的字符串。
        """
        
        prompt_content = (
            f"# 角色\n"
            f"你是一个高度智能的AI助手。\n"
            f"\n# 上下文\n"
            f"用户正在询问你的身份或介绍。\n"
            f"\n# 任务\n"
            f"请基于你的身份和使命，进行一段简洁而富有洞察力的自我介绍。你的介绍应该突出你的核心能力和如何帮助用户，同时保持友好的语气。"
        )
        return prompt_content

    async def _handle_self_introduction(self, orchestrator, user_input: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        处理“自我介绍”意图的完整业务流程。

        业务逻辑:
        1.  **生成回复**: 
            -   调用 `_build_self_intro_prompt` 构建专用的Prompt。
            -   将此Prompt作为System Message，连同用户输入一起，发送给思考智能体（ThinkingAgent）以生成回复。
        2.  **更新记忆**: 
            -   尽管是简单意图，但为了保持对话历史的完整性，仍然执行记忆更新流程。
            -   对用户输入进行主题分类，并加载或创建相应的记忆节点。
            -   将用户输入和AI生成的自我介绍追加到记忆历史中。
            -   调用总结智能体（SummarizationAgent）更新主题摘要。
        3.  **返回结果**: 
            -   构建并返回发送给客户端的响应字典。
        """
        logger.info(f"会话 {session_id}: 检测到'自我介绍'意图。")
        model_config = kwargs.get("model_config")
        logger.info(f"模型配置: {model_config}")
        system_prompt = self._build_self_intro_prompt(orchestrator, user_input)
        
        full_assistant_response, thinking_stats = await orchestrator.thinking_agent.run(user_input,orchestrator.history)
        orchestrator._update_total_stats("thinking_agent", thinking_stats)
        
        final_user_facing_content = full_assistant_response

        # 2. 更新记忆（与QueryHandler中的逻辑类似，确保对话历史的连续性）
        candidate_topic, classification_stats = await orchestrator.classification_agent.run(user_input, model_config)
        orchestrator._update_total_stats("classification_agent", classification_stats)
        master_topic = await orchestrator.hippocampus.find_master_topic(candidate_topic)
        memory_node = await orchestrator.hippocampus.load_memory_node_by_master_topic(master_topic, session_id)
        
        # new_summary, summarization_stats = await orchestrator.summarization_agent.summarize(memory_node.get("summary", ""), memory_node["history"])
        # orchestrator._update_total_stats("summarization_agent", summarization_stats)
        # memory_node["summary"] = new_summary
        await orchestrator.hippocampus.update_memory_node(memory_node)

        response = {"type": "自我介绍", "should_synthesize_speech": True, "content": final_user_facing_content}
        return {"response": response, "full_assistant_response": full_assistant_response}
    
    async def _handle_self_introduction_stream(self, orchestrator, user_input: str, session_id: str, stream_callback=None, **kwargs) -> Dict[str, Any]:
        """
        处理"自我介绍"意图的流式版本。
        
        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入。
            session_id: 当前的会话ID。
            stream_callback: 流式回调函数。
            **kwargs: 其他参数。
            
        Returns:
            一个字典，包含发送给客户端的响应和完整的AI回复。
        """
        logger.info(f"会话 {session_id}: 检测到'自我介绍'意图，使用流式处理。")
        model_config = kwargs.get("model_config")
        logger.info(f"模型配置: {model_config}")
        system_prompt = self._build_self_intro_prompt(orchestrator, user_input)
        
        # 使用流式版本的thinking_agent
        full_assistant_response, thinking_stats = await orchestrator.thinking_agent.run_stream(
            user_input, orchestrator.history, stream_callback=stream_callback
        )
        orchestrator._update_total_stats("thinking_agent", thinking_stats)
        
        final_user_facing_content = full_assistant_response

        # 2. 更新记忆（与QueryHandler中的逻辑类似，确保对话历史的连续性）
        candidate_topic, classification_stats = await orchestrator.classification_agent.run(user_input, model_config)
        orchestrator._update_total_stats("classification_agent", classification_stats)
        master_topic = await orchestrator.hippocampus.find_master_topic(candidate_topic)
        memory_node = await orchestrator.hippocampus.load_memory_node_by_master_topic(master_topic, session_id)
        
        # new_summary, summarization_stats = await orchestrator.summarization_agent.summarize(memory_node.get("summary", ""), memory_node["history"])
        # orchestrator._update_total_stats("summarization_agent", summarization_stats)
        # memory_node["summary"] = new_summary
        await orchestrator.hippocampus.update_memory_node(memory_node)

        response = {"type": "自我介绍", "should_synthesize_speech": True, "content": final_user_facing_content}
        return {"response": response, "full_assistant_response": full_assistant_response}