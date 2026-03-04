import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END

# --- 环境设置 ---
# 从 .env 或环境变量加载 API 密钥（请确保项目根目录 .env 中已配置 OPENAI_API_KEY 和 TAVILY_API_KEY）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# --- 定义工具 ---
# 研究员智能体将使用 Tavily 进行网络搜索
tool = TavilySearchResults(max_results=2)


# --- 定义智能体状态 ---
# 这是整个协作流程中共享的状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# --- 定义智能体节点 ---

# 这个函数将作为所有智能体节点的通用创建器
def create_agent_node(llm: ChatOpenAI, system_message: str):
    """
    创建一个智能体节点，该节点接收状态并返回更新后的消息列表。

    Args:
        llm: 用于生成响应的大语言模型。
        system_message: 定义智能体角色的系统提示。

    Returns:
        一个可用于图中的智能体节点函数。
    """
    prompt = PromptTemplate.from_template("""{system_message}

    当前对话内容:
    {messages}

    你的回应:""")
    agent = prompt | llm

    def agent_node(state: AgentState):
        response = agent.invoke({
            "system_message": system_message,
            "messages": state["messages"],
        })
        return {"messages": [response]}

    return agent_node


# --- 定义工具节点 ---

def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    执行工具调用并返回结果。

    Args:
        state: 当前图的状态。

    Returns:
        一个包含工具调用结果的消息字典。
    """
    # 获取最后一条消息中的工具调用
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    # 调用工具
    tool_outputs = []
    for call in tool_calls:
        # tool.invoke 会返回一个字符串，我们需要将其包装成 ToolMessage
        output = tool.invoke(call["args"])
        tool_outputs.append(
            {
                "tool_call_id": call["id"],
                "output": output,
                "name": call["name"],
            }
        )

    # 将工具输出包装成 AIMessage，以便模型可以理解
    from langchain_core.messages import AIMessage, ToolMessage

    # 创建 ToolMessage
    tool_messages = [
        ToolMessage(content=str(o["output"]), tool_call_id=o["tool_call_id"])
        for o in tool_outputs
    ]

    # 将其添加到 AIMessage 的 tool_calls 中
    # 这是为了让模型知道工具调用的结果
    response_message = AIMessage(
        content="",
        tool_calls=last_message.tool_calls,
    )
    response_message.tool_calls = [
        {**call, "output": str(output)}
        for call, output in zip(response_message.tool_calls, tool_outputs)
    ]

    return {"messages": tool_messages}


# --- 定义路由逻辑 ---

def router(state: AgentState) -> str:
    """
    决定下一个应该由哪个智能体或节点来处理。

    Args:
        state: 当前图的状态。

    Returns:
        下一个节点的名称。
    """
    last_message = state["messages"][-1]

    # 如果有工具调用，则路由到工具节点
    if last_message.tool_calls:
        return "tool_node"

    # 如果没有工具调用，并且发送者是研究员，则路由到作家
    if state["sender"] == "researcher":
        return "writer_node"

    # 否则，结束流程
    return "__end__"


# --- 构建图 ---

# 1. 初始化模型 - 使用 OpenRouter 配置
import sys
sys.path.append('/home/gmm/srv/cognitive-center')
from cognitive_graph.config import config

llm = ChatOpenAI(
    model=config.OPENROUTER_MODEL,
    openai_api_base=config.OPENROUTER_BASE_URL,
    openai_api_key=config.OPENROUTER_API_KEY
)

# 2. 创建智能体节点
researcher_llm = llm.bind_tools([tool])
researcher_node = create_agent_node(
    researcher_llm,
    "你是一个专业的研究员。你的任务是找到用户问题的答案，并提供详细的来源和信息。",
)
writer_node = create_agent_node(
    llm,
    "你是一个专业的作家。你的任务是根据研究员提供的信息，撰写一篇清晰、简洁、引人入胜的文章。",
)

# 3. 初始化 StateGraph
graph = StateGraph(AgentState)

# 4. 添加节点
graph.add_node("researcher_node", researcher_node)
graph.add_node("writer_node", writer_node)
graph.add_node("tool_node", tool_node)

# 5. 设置入口点
graph.set_entry_point("researcher_node")

# 6. 添加边
graph.add_conditional_edges(
    "researcher_node",
    router,
    {"writer_node": "writer_node", "tool_node": "tool_node", "__end__": "__end__"},
)
graph.add_edge("writer_node", END) # 作家节点之后总是结束
graph.add_edge("tool_node", "researcher_node") # 工具执行完后返回给研究员

# 7. 编译图
app = graph.compile()


# --- 运行图 ---

def run_multi_agent_collaboration(query: str):
    """
    使用给定的查询运行多智能体协作流程。

    Args:
        query: 用户输入的查询字符串。
    """
    inputs = {
        "messages": [HumanMessage(content=query)],
        "sender": "researcher",
    }
    # 流式输出结果
    for output in app.stream(inputs, stream_mode="values"):
        # stream_mode="values" 会直接返回状态的值，而不是整个状态字典
        print("--- 智能体输出 ---")
        print(output["messages"][-1].content)


if __name__ == "__main__":
    # print("--- 示例: 研究 LangGraph 并撰写一篇关于它的文章 ---")
    # run_multi_agent_collaboration("LangGraph 是什么？它与 LangChain 的主要区别是什么？")

    print("--- 示例 ---")
    run_multi_agent_collaboration("小米今天股价是多少？")