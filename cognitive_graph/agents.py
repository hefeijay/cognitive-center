#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体协作模块
基于LangGraph实现的多智能体协作框架，支持持续多轮对话
"""

import json
import uuid
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from datetime import datetime, timezone

import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .database import DatabaseManager
from .config import config


# --- 定义智能体状态 ---
class AgentState(TypedDict):
    """智能体协作状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    current_agent: str
    message_id: str
    conversation_round: int
    context: Dict[str, Any]


class MultiAgentCollaborationFramework:
    """多智能体协作框架"""
    
    def __init__(self):
        # 使用OpenRouter配置
        self.llm = ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            openai_api_base=config.OPENROUTER_BASE_URL,
            openai_api_key=config.OPENROUTER_API_KEY,
            temperature=0.7
        )
        self.db_manager = DatabaseManager()
        self.graph = None
        self.agents = {}
        self._build_graph()
    
    def _load_agent_prompts(self) -> Dict[str, str]:
        """从数据库加载智能体prompt"""
        prompts = {}
        try:
            with DatabaseManager() as db:
                agent_prompts = db.get_all_agent_prompts("graph_agent")
                print(f"代理模版:{agent_prompts}")
                for prompt in agent_prompts:
                    prompts[prompt.agent_name] = prompt.template
        except Exception as e:
            print(f"加载智能体prompt失败: {e}")
            # 使用默认prompt
            prompts = self._get_default_prompts()
        
        return prompts
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """获取默认的智能体prompt"""
        return {
            "analyzer": """你是一个专业的数据分析师。你的任务是：
1. 分析接收到的消息内容
2. 识别关键信息和模式
3. 提供初步的分析结果
4. 为后续处理提供建议

当前消息内容: {content}
对话历史: {history}

请提供你的分析结果：""",
            
            "decision_maker": """你是一个决策专家。基于分析师的分析结果，你需要：
1. 评估当前情况
2. 制定具体的行动建议
3. 确定优先级和紧急程度
4. 提供可执行的决策方案

分析结果: {analysis}
对话历史: {history}

请提供你的决策建议：""",
            
            "executor": """你是一个执行专家。基于决策建议，你需要：
1. 制定具体的执行计划
2. 生成标准化的决策消息
3. 确保消息格式符合要求
4. 提供执行状态反馈

决策建议: {decision}
对话历史: {history}

请生成最终的决策消息："""
        }
    
    def _create_agent_node(self, agent_name: str, system_prompt: str):
        """创建智能体节点"""
        def agent_node(state: AgentState):
            # 构建消息历史
            history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"][-5:]])
            
            # 根据智能体类型准备不同的输入
            if agent_name == "analyzer":
                # 获取原始消息内容
                original_content = state["context"].get("original_content", "")
                prompt_input = {
                    "content": original_content,
                    "history": history
                }
            elif agent_name == "decision_maker":
                # 获取分析结果
                analysis = state["messages"][-1].content if state["messages"] else ""
                prompt_input = {
                    "analysis": analysis,
                    "history": history
                }
            elif agent_name == "executor":
                # 获取决策建议
                decision = state["messages"][-1].content if state["messages"] else ""
                prompt_input = {
                    "decision": decision,
                    "history": history
                }
            else:
                prompt_input = {"history": history}
            
            # 创建prompt模板并调用LLM
            prompt = PromptTemplate.from_template(system_prompt)
            response = self.llm.invoke([
                SystemMessage(content=prompt.format(**prompt_input))
            ])
            
            # 保存对话历史到数据库
            try:
                with DatabaseManager() as db:
                    db.save_chat_message(
                        session_id=state["session_id"],
                        role=agent_name,
                        content=response.content,
                        message_type="agent_response",
                        message_id=str(uuid.uuid4())
                    )
            except Exception as e:
                print(f"保存对话历史失败: {e}")
            
            return {
                "messages": [response],
                "current_agent": agent_name,
                "conversation_round": state["conversation_round"] + 1
            }
        
        return agent_node
    
    def _router(self, state: AgentState) -> str:
        """路由逻辑，决定下一个智能体"""
        current_agent = state.get("current_agent", "")
        conversation_round = state.get("conversation_round", 0)
        
        # 检查是否超过最大轮次
        if conversation_round >= config.MAX_CONVERSATION_ROUNDS:
            return "__end__"
        
        # 智能体流转逻辑
        if current_agent == "" or current_agent == "start":
            return "analyzer"
        elif current_agent == "analyzer":
            return "decision_maker"
        elif current_agent == "decision_maker":
            return "executor"
        else:
            return "__end__"
    
    def _build_graph(self):
        """构建智能体协作图"""
        # 加载智能体prompt
        prompts = self._load_agent_prompts()
        print(f"加载提示词：{prompts}")
        # 创建智能体节点
        for agent_name, prompt in prompts.items():
            self.agents[agent_name] = self._create_agent_node(agent_name, prompt)
        
        # 如果没有从数据库加载到prompt，使用默认的
        if not self.agents:
            default_prompts = self._get_default_prompts()
            for agent_name, prompt in default_prompts.items():
                self.agents[agent_name] = self._create_agent_node(agent_name, prompt)
        
        # 初始化StateGraph
        self.graph = StateGraph(AgentState)
        
        # 添加节点
        for agent_name, agent_node in self.agents.items():
            self.graph.add_node(agent_name, agent_node)
        
        # 设置入口点
        self.graph.set_entry_point("analyzer")
        
        # 添加条件边
        for agent_name in self.agents.keys():
            self.graph.add_conditional_edges(
                agent_name,
                self._router,
                {next_agent: next_agent for next_agent in self.agents.keys()} | {"__end__": "__end__"}
            )
        
        # 编译图
        self.graph = self.graph.compile()
    
    def process_message(self, message_id: str) -> Optional[str]:
        """处理单个消息"""
        try:
            with DatabaseManager() as db:
                # 获取消息
                messages = db.get_pending_messages(limit=1)
                if not messages:
                    return None
                
                message = messages[0]
                
                # 更新消息状态为处理中
                db.update_message_status(
                    message.message_id, 
                    'processing',
                    consumed_at=datetime.now(timezone.utc)
                )
                
                # 获取或创建会话ID
                session_id = f"session_{message.message_id}"
                
                # 加载对话历史
                chat_history = db.get_chat_history(session_id)
                history_messages = []
                for chat in chat_history:
                    if chat.role == "user":
                        history_messages.append(HumanMessage(content=chat.content))
                    else:
                        history_messages.append(AIMessage(content=chat.content))
                
                # 添加当前消息
                current_message = HumanMessage(content=message.content)
                history_messages.append(current_message)
                
                # 保存用户消息到对话历史
                db.save_chat_message(
                    session_id=session_id,
                    role="user",
                    content=message.content,
                    message_type="user_input",
                    message_id=message.message_id
                )
                
                # 构建初始状态
                initial_state = {
                    "messages": history_messages,
                    "session_id": session_id,
                    "current_agent": "start",
                    "message_id": message.message_id,
                    "conversation_round": 0,
                    "context": {
                        "original_content": message.content,
                        "message_type": message.message_type,
                        "metadata": message.metadata
                    }
                }
                
                # 运行智能体协作
                final_decision = None
                for output in self.graph.stream(initial_state):
                    if "__end__" in output:
                        # 获取最终决策
                        if "messages" in output["__end__"] and output["__end__"]["messages"]:
                            final_decision = output["__end__"]["messages"][-1].content
                        break
                
                # 如果没有获得最终决策，从最后的消息中获取
                if not final_decision and "executor" in output:
                    executor_output = output["executor"]
                    if "messages" in executor_output and executor_output["messages"]:
                        final_decision = executor_output["messages"][-1].content
                
                # 保存AI决策到数据库
                if final_decision:
                    decision_id = f"decision_{uuid.uuid4()}"
                    db.create_ai_decision(
                        decision_id=decision_id,
                        decision_type="analysis",
                        message=final_decision,
                        source="multi_agent_system",
                        source_id=message.message_id,
                        priority=message.priority,
                        confidence=85.0
                    )
                
                # 更新消息状态为已完成
                db.update_message_status(
                    message.message_id,
                    'completed',
                    completed_at=datetime.now(timezone.utc)
                )
                
                return final_decision
                
        except Exception as e:
            # 更新消息状态为失败
            try:
                with DatabaseManager() as db:
                    db.update_message_status(
                        message_id,
                        'failed',
                        error_message=str(e)
                    )
            except:
                pass
            raise e
    
    def run_continuous_processing(self):
        """持续处理消息队列"""
        print("开始持续处理消息队列...")
        
        while True:
            try:
                with DatabaseManager() as db:
                    pending_messages = db.get_pending_messages(config.MESSAGE_BATCH_SIZE)
                    
                    if not pending_messages:
                        print("没有待处理的消息，等待中...")
                        import time
                        time.sleep(config.MESSAGE_POLL_INTERVAL)
                        continue
                    
                    for message in pending_messages:
                        print(f"处理消息: {message.message_id}")
                        try:
                            result = self.process_message(message.message_id)
                            if result:
                                print(f"消息处理完成: {message.message_id}")
                                print(f"决策结果: {result[:100]}...")
                            else:
                                print(f"消息处理失败: {message.message_id}")
                        except Exception as e:
                            print(f"处理消息 {message.message_id} 时出错: {e}")
                            continue
                    
            except KeyboardInterrupt:
                print("收到停止信号，退出处理循环")
                break
            except Exception as e:
                print(f"处理循环出错: {e}")
                import time
                time.sleep(config.MESSAGE_POLL_INTERVAL)