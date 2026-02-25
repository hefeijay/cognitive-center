# cognitive_model/handlers/tuning_handler.py

import logging
import json
from .base_handler import BaseHandler
from ..tools.native.file_editor import read_file, write_file
from cognitive_model.config.prompt_manager import PromptManager
from typing import Dict, Any, Optional, List

logger = logging.getLogger("app.handler.tuning")

class TuningHandler(BaseHandler):
    """
    处理“调整”意图和后续的批准/拒绝流程。

    这个处理器是实现AI自我认知和修正能力的核心。它管理一个完整的、
    涉及多轮对话的交互流程：从理解用户的调整需求，到生成具体的修正计划，
    再到请求用户批准，并最终执行高权限的文件写入操作来更新自身的行为准则（Prompts）。
    """
    def __init__(self, prompt_manager: PromptManager):
        """
        初始化分类智能体。

        Args:
            prompt_manager: PromptManager 的一个实例。
        """
        self.prompt_manager = prompt_manager
    
    async def handle_stream(self, orchestrator, user_input: str, session_id: str, stream_callback=None, **kwargs) -> Dict[str, Any]:
        """
        处理"调整"意图的流式版本。
        
        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入。
            session_id: 当前的会话ID。
            stream_callback: 流式回调函数，用于实时发送响应片段。
            **kwargs: 其他参数。
            
        Returns:
            一个字典，包含发送给客户端的响应和完整的AI回复。
        """
        state_manager = orchestrator.session_state_manager
        pending_plan = state_manager.get_state(session_id, "pending_update_plan")

        if pending_plan:
            return await self._handle_pending_approval_stream(orchestrator, user_input, session_id, pending_plan, stream_callback)
        else:
            return await self._handle_tune_intent_stream(orchestrator, user_input, session_id, stream_callback)
        
    async def handle(self, orchestrator, user_input: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        根据会话状态，决定是启动一个新的调整流程还是处理一个待批准的计划。

        业务逻辑:
        1.  使用 `SessionStateManager` 检查当前会话是否存在一个名为 `pending_update_plan` 的状态。
        2.  如果存在，说明系统正在等待用户对一个已提出的修改计划进行审批。此时，调用 `_handle_pending_approval` 方法处理用户的“是/否”回答。
        3.  如果不存在，说明这是一个新的“调整”请求。此时，调用 `_handle_tune_intent` 方法启动一个新的认知修正流程。
        """
        state_manager = orchestrator.session_state_manager
        pending_plan = state_manager.get_state(session_id, "pending_update_plan")

        if pending_plan:
            return await self._handle_pending_approval(orchestrator, user_input, session_id, pending_plan)
        else:
            return await self._handle_tune_intent(orchestrator, user_input, session_id)

    async def _handle_tune_intent(self, orchestrator, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        处理“调整”意图的初始请求，生成并向用户提出一个具体的修改计划。

        业务逻辑:
        1.  调用 `CognitiveTunerAgent`，将用户的反馈（user_input）传递给它，以生成一个详细的、结构化的修改计划（update_plan）。
        2.  如果计划生成失败（例如，LLM无法理解或拒绝修改），则向用户返回错误信息。
        3.  将成功生成的 `update_plan` 存储在当前会话的状态中，键为 `pending_update_plan`。这使得系统可以在用户的下一条消息中继续此流程。
        4.  格式化 `update_plan` 中的分析内容，构建一段清晰、礼貌的文本，向用户解释即将进行的修改，并请求其批准。
        """
        logger.info(f"会话 {session_id}: 检测到'调整'意图，启动认知修正流程...")
        
        tuner_agent = orchestrator.cognitive_tuner_agent
        state_manager = orchestrator.session_state_manager

        update_plan, tuner_stats = await tuner_agent.generate_update_plan(user_input)
        orchestrator._update_total_stats("cognitive_tuner_agent", tuner_stats)

        if "error" in update_plan:
            response = {"type": "错误", "should_synthesize_speech": True, "content": update_plan["error"]}
            return {"response": response, "full_assistant_response": update_plan["error"]}

        state_manager.set_state(session_id, "pending_update_plan", update_plan)

        analysis = update_plan.get('analysis', '无分析')
        response_text = (
            f"好的，我理解了。根据您的反馈，我生成了一个自我修正计划：\n\n"
            f"【我的分析】\n{analysis}\n\n"
            f"我将对我的内部Prompt进行修改。请问您是否批准执行此项修改？\n"
            f"请回答“是”或“否”。"
        )
        
        response = {"type": "调整", "should_synthesize_speech": True, "content": response_text}
        return {"response": response, "full_assistant_response": response_text}

    async def _handle_pending_approval(self, orchestrator, user_input: str, session_id: str, update_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理用户对一个待审批计划的回复（批准或拒绝）。

        业务逻辑:
        1.  首先，无论用户如何回答，都立即从会话状态中清理 `pending_update_plan`，因为这个一次性的审批流程已经结束。
        2.  检查用户的输入是否表示“同意”（例如“是”、“可以”、“yes”等）。
        3.  **如果用户批准**:
            a.  记录一条高权限操作的警告日志。
            b.  从 `update_plan` 中提取目标文件路径、目标Agent、目标Prompt键和新的Prompt内容。
            c.  读取目标JSON配置文件，定位到需要修改的具体位置，并用新Prompt替换旧内容。
            d.  将修改后的数据写回文件系统，这是一个高风险操作。
            e.  如果写入成功，重新加载 `prompt_manager` 使修改立即生效，并通知用户成功。
            f.  如果在任何步骤中发生异常，捕获错误并向用户报告。
        4.  **如果用户拒绝**:
            a.  记录用户拒绝操作的日志。
            b.  向用户发送一条消息，确认操作已取消。
        """
        logger.info(f"会话 {session_id}: 检测到待审批的修改计划，正在处理用户回复...")
        
        state_manager = orchestrator.session_state_manager
        state_manager.clear_state(session_id, "pending_update_plan")
        
        if user_input.strip().lower() in ["是", "好的", "可以", "批准", "同意", "yes", "ok", "y"]:
            logger.warning(f"会话 {session_id}: 用户已批准修改计划，即将执行高权限文件写入操作！")
            
            try:
                target_file_key = update_plan['target_file_key']
                prompts_json_str = read_file(target_file_key)
                prompts_data = json.loads(prompts_json_str)
                
                target_agent = update_plan['target_agent']
                target_key = update_plan['target_prompt_key']
                new_prompt = update_plan['proposed_new_prompt']
                
                if target_key:
                    prompts_data[target_agent]["templates"][target_key] = new_prompt
                else:
                    prompts_data[target_agent]["template"] = new_prompt
                
                updated_content_str = json.dumps(prompts_data, ensure_ascii=False, indent=2)
                write_result = write_file(target_file_key, updated_content_str)
                
                if write_result.startswith("错误："):
                    content = f"在执行修改时发生错误：{write_result}"
                else:
                    self.prompt_manager.load_prompts()
                    content = "修改已成功应用。我的新行为模式将在下一次对话中生效。"

            except Exception as e:
                logger.exception(f"执行修改计划时发生严重错误: {e}")
                content = f"抱歉，在应用修改时发生了意外的内部错误：{e}"

            response = {"type": "调整", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}
        else:
            logger.info(f"会话 {session_id}: 用户已拒绝修改计划。")
            content = "好的，操作已取消。我将保持现有的行为模式。"
            response = {"type": "调整", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}
    
    async def _handle_tune_intent_stream(self, orchestrator, user_input: str, session_id: str, stream_callback=None) -> Dict[str, Any]:
        """
        处理"调整"意图初始请求的流式版本。
        
        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入。
            session_id: 当前的会话ID。
            stream_callback: 流式回调函数。
            
        Returns:
            一个字典，包含发送给客户端的响应和完整的AI回复。
        """
        logger.info(f"会话 {session_id}: 检测到'调整'意图，启动认知修正流程（流式）...")
        
        tuner_agent = orchestrator.cognitive_tuner_agent
        state_manager = orchestrator.session_state_manager

        update_plan, tuner_stats = await tuner_agent.generate_update_plan(user_input)
        orchestrator._update_total_stats("cognitive_tuner_agent", tuner_stats)

        if "error" in update_plan:
            if stream_callback:
                await stream_callback(update_plan["error"], "content")
            response = {"type": "错误", "should_synthesize_speech": True, "content": update_plan["error"]}
            return {"response": response, "full_assistant_response": update_plan["error"]}

        state_manager.set_state(session_id, "pending_update_plan", update_plan)

        analysis = update_plan.get('analysis', '无分析')
        response_text = (
            f"好的，我理解了。根据您的反馈，我生成了一个自我修正计划：\n\n"
            f"【我的分析】\n{analysis}\n\n"
            f"我将对我的内部Prompt进行修改。请问您是否批准执行此项修改？\n"
            f"请回答\"是\"或\"否\"。"
        )
        
        # 流式发送响应
        if stream_callback:
            await stream_callback(response_text, "content")
        
        response = {"type": "调整", "should_synthesize_speech": True, "content": response_text}
        return {"response": response, "full_assistant_response": response_text}
    
    async def _handle_pending_approval_stream(self, orchestrator, user_input: str, session_id: str, update_plan: Dict[str, Any], stream_callback=None) -> Dict[str, Any]:
        """
        处理用户对待审批计划回复的流式版本。
        
        Args:
            orchestrator: 认知协调器实例。
            user_input: 用户的原始输入。
            session_id: 当前的会话ID。
            update_plan: 待审批的更新计划。
            stream_callback: 流式回调函数。
            
        Returns:
            一个字典，包含发送给客户端的响应和完整的AI回复。
        """
        logger.info(f"会话 {session_id}: 检测到待审批的修改计划，正在处理用户回复（流式）...")
        
        state_manager = orchestrator.session_state_manager
        state_manager.clear_state(session_id, "pending_update_plan")
        
        if user_input.strip().lower() in ["是", "好的", "可以", "批准", "同意", "yes", "ok", "y"]:
            logger.warning(f"会话 {session_id}: 用户已批准修改计划，即将执行高权限文件写入操作！")
            
            try:
                target_file_key = update_plan['target_file_key']
                prompts_json_str = read_file(target_file_key)
                prompts_data = json.loads(prompts_json_str)
                
                target_agent = update_plan['target_agent']
                target_key = update_plan['target_prompt_key']
                new_prompt = update_plan['proposed_new_prompt']
                
                if target_key:
                    prompts_data[target_agent]["templates"][target_key] = new_prompt
                else:
                    prompts_data[target_agent]["template"] = new_prompt
                
                updated_content_str = json.dumps(prompts_data, ensure_ascii=False, indent=2)
                write_result = write_file(target_file_key, updated_content_str)
                
                if write_result.startswith("错误："):
                    content = f"在执行修改时发生错误：{write_result}"
                else:
                    self.prompt_manager.load_prompts()
                    content = "修改已成功应用。我的新行为模式将在下一次对话中生效。"

            except Exception as e:
                logger.exception(f"执行修改计划时发生严重错误: {e}")
                content = f"抱歉，在应用修改时发生了意外的内部错误：{e}"

            # 流式发送响应
            if stream_callback:
                await stream_callback(content, "content")
            
            response = {"type": "调整", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}
        else:
            logger.info(f"会话 {session_id}: 用户已拒绝修改计划。")
            content = "好的，操作已取消。我将保持现有的行为模式。"
            
            # 流式发送响应
            if stream_callback:
                await stream_callback(content)
            
            response = {"type": "调整", "should_synthesize_speech": True, "content": content}
            return {"response": response, "full_assistant_response": content}