import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Annotated, List, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- 环境设置 ---
# 从 .env 或环境变量加载 OPENAI_API_KEY（请确保项目根目录 .env 中已配置）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# --- 定义工具 ---
# 初始化 DuckDuckGo 搜索工具
# DuckDuckGo 是一个无需 API 密钥即可使用的搜索引擎
search_tool = DuckDuckGoSearchRun()

# 将所有工具放入一个列表中
tools = [search_tool]


# --- 定义图的状态 ---
# 这个状态会贯穿整个图的执行过程
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# --- 初始化模型 ---
# 我们将使用 OpenAI 的 gpt-4o 模型
# 并将工具绑定到模型上，让模型知道有哪些工具可用
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# --- 定义图的节点 ---

# 1. 智能体节点 (Agent)
#    - 负责调用大语言模型
#    - 根据用户输入和当前状态，决定是直接回答还是调用工具
def call_model(state: AgentState):
    """
    调用大语言模型来处理用户输入。

    Args:
        state: 当前图的状态。

    Returns:
        一个包含模型响应的消息字典。
    """
    messages = state["messages"]
    response = model.invoke(messages)
    # 将模型的响应添加到状态中
    return {"messages": [response]}


# 2. 工具节点 (Tool Node)
#    - 负责执行模型请求的工具调用
#    - ToolNode 是 LangGraph 提供的一个预构建节点，可以简化工具执行的过程
tool_node = ToolNode(tools)


# --- 定义图的边 ---

# 条件边 (Conditional Edge)
# - 决定在智能体节点之后，是调用工具、结束流程，还是重新规划
def should_continue(state: AgentState) -> str:
    """
    根据模型的最后一条消息中是否包含工具调用来决定下一步的操作。

    Args:
        state: 当前图的状态。

    Returns:
        - "tools": 如果需要调用工具
        - "end": 如果流程可以结束
    """
    last_message = state["messages"][-1]
    # 如果没有工具调用，流程结束
    if not last_message.tool_calls:
        return "end"
    # 否则，调用工具
    return "tools"


# --- 构建图 ---

# 1. 初始化一个 StateGraph，并传入我们定义的状态对象
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 3. 设置入口点
workflow.set_entry_point("agent")

# 4. 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": "__end__",
    },
)

# 5. 添加从工具节点返回到智能体节点的边
workflow.add_edge("tools", "agent")

# 6. 编译图，生成可执行的应用
app = workflow.compile()


# --- 运行图 ---

def run_web_search_agent(query: str):
    """
    使用给定的查询运行网络搜索智能体。

    Args:
        query: 用户输入的查询字符串。
    """
    inputs = {"messages": [HumanMessage(content=query)]}
    # 流式输出结果，可以清晰地看到每一步的执行过程和结果
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"--- 输出自节点: '{key}' ---")
            print(value)
        print("\n---")


if __name__ == "__main__":
    # print("--- 示例: 查询 LangGraph 的最新版本 ---")
    # run_web_search_agent("langgraph的最新版本是多少？")

    # print("\n--- 示例: 查询天气 ---")
    # run_web_search_agent("今天北京的天气怎么样？")

    print("\n--- 示例 ---")
    run_web_search_agent("筑波今天的天气是多少？")