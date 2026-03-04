import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 从 .env 或环境变量加载 OPENAI_API_KEY（请确保项目根目录 .env 中已配置）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# 定义一个简单的工具，用于加法计算
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


# 定义图的状态
# 这个状态会贯穿整个图的执行过程
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# 初始化模型
# 使用 OpenRouter 配置
import sys
sys.path.append('/home/gmm/srv/cognitive-center')
from cognitive_graph.config import config

llm = ChatOpenAI(
    model=config.OPENROUTER_MODEL,
    openai_api_base=config.OPENROUTER_BASE_URL,
    openai_api_key=config.OPENROUTER_API_KEY
)

# 将模型和工具绑定，让模型知道有哪些工具可用
llm_with_tools = llm.bind_tools([add])

# 定义智能体节点
# 这个节点负责调用大语言模型，并根据用户的输入决定是否要调用工具
def agent_node(state: State):
    """
    调用大语言模型来处理用户输入。

    Args:
        state: 当前图的状态。

    Returns:
        一个包含模型响应的消息字典。
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 定义工具节点
# 这个节点负责执行工具调用
tool_node = ToolNode([add])


# 定义条件边
# 这条边将决定在智能体节点之后，是调用工具还是结束流程
def should_continue(state: State) -> str:
    """
    根据模型的最后一条消息中是否包含工具调用来决定下一步的操作。

    Args:
        state: 当前图的状态。

    Returns:
        如果需要调用工具，则返回 "tools"；否则返回 "end"。
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


# 构建图
graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

# 设置入口点
graph_builder.set_entry_point("agent")

# 添加条件边
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": "__end__"},
)

# 从工具节点返回到智能体节点
graph_builder.add_edge("tools", "agent")

# 编译图
graph = graph_builder.compile()


# 定义一个函数来运行图
def run_graph(input_message: str):
    """
    使用给定的输入消息运行图。

    Args:
        input_message: 用户输入的字符串。
    """
    # 流式输出图的执行过程
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=input_message)]},
        # 我们可以在这里设置断点，以便在每个步骤后进行调试
        # interrupt_after=["agent", "tools"],
    ):
        # 打印每个步骤的输出
        for key, value in chunk.items():
            print(f"--- {key} ---")
            print(value)
        print("---")


if __name__ == "__main__":
    # 示例 1: 一个简单的问候
    print("--- 示例 1: 简单问候 ---")
    run_graph("你好！")

    # 示例 2: 调用工具进行计算
    print("\n--- 示例 2: 调用工具进行计算 ---")
    run_graph("3加5等于多少？")

    # 示例 3: 一个更复杂的问题，可能需要多次思考
    # 注意：这个简单的智能体可能无法完美解决，但展示了循环的可能性
    print("\n--- 示例 3: 需要思考的复杂问题 ---")
    run_graph("我口袋里有两枚硬币，总共是7角钱。其中一枚不是5角，请问这两枚硬币的面值是多少？")