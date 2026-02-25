#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP服务集成模块
将第三方MCP服务集成为智能体工具
"""

import json
import requests
from typing import Any, Dict, List, Optional, Union
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .config import config


class MCPToolInput(BaseModel):
    """MCP工具输入模型"""
    method: str = Field(description="MCP方法名")
    params: Dict[str, Any] = Field(default_factory=dict, description="方法参数")


class MCPTool(BaseTool):
    """MCP服务工具"""
    
    name: str = "mcp_service"
    description: str = "调用MCP服务执行各种操作，如数据查询、文件操作、系统管理等"
    args_schema: type[BaseModel] = MCPToolInput
    
    def __init__(self, mcp_server_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url or config.MCP_SERVER_URL
    
    def _run(self, method: str, params: Dict[str, Any] = None) -> str:
        """执行MCP服务调用"""
        try:
            # 构建MCP请求
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or {}
            }
            
            # 发送请求到MCP服务器
            response = requests.post(
                f"{self.mcp_server_url}/mcp",
                json=mcp_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    return json.dumps(result["result"], ensure_ascii=False, indent=2)
                elif "error" in result:
                    return f"MCP服务错误: {result['error']}"
                else:
                    return "MCP服务返回了未知格式的响应"
            else:
                return f"MCP服务请求失败，状态码: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "MCP服务请求超时"
        except requests.exceptions.ConnectionError:
            return "无法连接到MCP服务器"
        except Exception as e:
            return f"调用MCP服务时出错: {str(e)}"
    
    async def _arun(self, method: str, params: Dict[str, Any] = None) -> str:
        """异步执行MCP服务调用"""
        # 这里可以使用aiohttp实现异步版本
        return self._run(method, params)


class MCPServiceManager:
    """MCP服务管理器"""
    
    def __init__(self, mcp_server_url: str = None):
        self.mcp_server_url = mcp_server_url or config.MCP_SERVER_URL
        self.available_tools = []
        self._discover_tools()
    
    def _discover_tools(self):
        """发现可用的MCP工具"""
        try:
            # 调用MCP服务的tools/list方法获取可用工具
            response = requests.post(
                f"{self.mcp_server_url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "tools" in result["result"]:
                    self.available_tools = result["result"]["tools"]
                    print(f"发现 {len(self.available_tools)} 个MCP工具")
                else:
                    print("MCP服务未返回工具列表")
            else:
                print(f"获取MCP工具列表失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"发现MCP工具时出错: {e}")
    
    def get_mcp_tool(self) -> MCPTool:
        """获取MCP工具实例"""
        return MCPTool(mcp_server_url=self.mcp_server_url)
    
    def call_mcp_method(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """直接调用MCP方法"""
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or {}
            }
            
            response = requests.post(
                f"{self.mcp_server_url}/mcp",
                json=mcp_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": {
                        "code": response.status_code,
                        "message": f"HTTP错误: {response.status_code}"
                    }
                }
                
        except Exception as e:
            return {
                "error": {
                    "code": -1,
                    "message": f"调用MCP方法时出错: {str(e)}"
                }
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        return self.available_tools


# 创建全局MCP服务管理器实例
mcp_manager = MCPServiceManager()


def create_mcp_enhanced_agent_node(agent_name: str, system_prompt: str, llm, use_mcp: bool = True):
    """创建支持MCP工具的智能体节点"""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.prompts import PromptTemplate
    
    def agent_node(state):
        # 构建消息历史
        history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"][-5:]])
        
        # 准备prompt输入
        if agent_name == "analyzer":
            original_content = state["context"].get("original_content", "")
            prompt_input = {
                "content": original_content,
                "history": history
            }
        elif agent_name == "decision_maker":
            analysis = state["messages"][-1].content if state["messages"] else ""
            prompt_input = {
                "analysis": analysis,
                "history": history
            }
        elif agent_name == "executor":
            decision = state["messages"][-1].content if state["messages"] else ""
            prompt_input = {
                "decision": decision,
                "history": history
            }
        else:
            prompt_input = {"history": history}
        
        # 如果启用MCP工具，添加工具信息到prompt
        if use_mcp and mcp_manager.available_tools:
            tools_info = "\n可用的MCP工具:\n"
            for tool in mcp_manager.available_tools[:5]:  # 限制显示前5个工具
                tools_info += f"- {tool.get('name', 'unknown')}: {tool.get('description', 'no description')}\n"
            
            enhanced_prompt = system_prompt + "\n\n" + tools_info + "\n如果需要，你可以建议使用这些工具来辅助分析和决策。"
        else:
            enhanced_prompt = system_prompt
        
        # 创建prompt模板并调用LLM
        prompt = PromptTemplate.from_template(enhanced_prompt)
        
        # 如果启用MCP并且LLM支持工具调用
        if use_mcp:
            try:
                # 绑定MCP工具到LLM
                mcp_tool = mcp_manager.get_mcp_tool()
                llm_with_tools = llm.bind_tools([mcp_tool])
                response = llm_with_tools.invoke([
                    SystemMessage(content=prompt.format(**prompt_input))
                ])
            except Exception as e:
                print(f"使用MCP工具时出错: {e}，回退到普通模式")
                response = llm.invoke([
                    SystemMessage(content=prompt.format(**prompt_input))
                ])
        else:
            response = llm.invoke([
                SystemMessage(content=prompt.format(**prompt_input))
            ])
        
        return {
            "messages": [response],
            "current_agent": agent_name,
            "conversation_round": state["conversation_round"] + 1
        }
    
    return agent_node