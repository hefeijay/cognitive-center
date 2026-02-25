# 认知模型 (Cognitive Model) 模块详解

本模块是整个认知中心的核心，负责实现和管理所有与AI认知、决策、记忆和工具使用相关的功能。它被设计为一个高度模块化、可扩展的系统，能够模拟复杂的认知过程。

## 核心设计理念

- **认知架构 (Cognitive Architecture)**: 模仿人类的认知流程，将复杂的任务分解为意图识别、信息检索、思考推理、工具调用和答案生成等多个阶段。
- **模块化 (Modularity)**: 每个子模块（如 `agents`, `handlers`, `hippocampus`）都有明确的职责，易于独立开发、测试和维护。
- **动态自适应 (Dynamic Adaptation)**: 通过 `Orchestrator` 和 `Handler` 的组合，系统能够根据用户意图动态选择最合适的处理流程和AI智能体。
- **状态与记忆 (State and Memory)**: 通过 `hippocampus` 模块，系统能够管理长期和短期的记忆，从而在持续的对话中保持上下文连贯性。

## 模块结构树

```
cognitive_model/
│
├── __init__.py
├── orchestrator.py             # 核心协调器，认知流程的总指挥
│
├── agents/                     # 存放各类专业化AI智能体
│   ├── __init__.py
│   ├── classification_agent.py   # 分类智能体
│   ├── cognitive_tuner_agent.py  # 认知调优智能体
│   ├── formatting_agent.py       # 格式化智能体
│   ├── intent_agent.py           # 意图识别智能体
│   ├── llm_utils.py              # 大语言模型工具类
│   ├── routing_agent.py          # 路由决策智能体
│   ├── summarization_agent.py    # 摘要生成智能体
│   ├── thinking_agent.py         # 思考与推理智能体
│   └── topic_matching_agent.py   # 主题匹配智能体
│
├── config/                     # 配置模块
│   └── prompt_manager.py         # 提示词管理器
│
├── handlers/                   # 意图处理器
│   ├── __init__.py
│   ├── base_handler.py           # 处理器基类
│   ├── query_handler.py          # "提问"意图处理器
│   ├── session_state_manager.py  # 会话状态管理器
│   ├── simple_handler.py         # "简单"意图处理器
│   └── tuning_handler.py         # "调优"意图处理器
│
├── hippocampus/                # "海马体"，负责记忆管理
│   ├── __init__.py
│   ├── handler.py                # 记忆处理器
│   ├── memory_storage/           # 长期记忆存储
│   ├── session_handler.py        # 会话记忆处理器
│   └── session_storage/          # 短期会话存储
│
├── knowledge/                  # 知识库（未来扩展）
│
├── tasks/                      # 异步任务管理
│   ├── __init__.py
│   ├── task_handler.py           # 任务处理器
│   └── task_storage/             # 任务存储
│
└── tools/                      # AI智能体可用的工具
    ├── __init__.py
    ├── native/                   # 本地工具
    │   ├── __init__.py
    │   ├── calculator.py         # 计算器
    │   ├── file_analyzer.py      # 文件分析器
    │   └── file_editor.py        # 文件编辑器
    └── tool_registry.py          # 工具注册表
```

## 核心组件详解

### 1. `orchestrator.py` - 核心协调器

- **职责**: 作为认知模型的大脑，接收来自上层应用（如 `singa_one_server`）的请求。
- **流程**:
    1. 调用 `IntentAgent` 识别用户的主要意图（如 "提问"、"请求帮助"）。
    2. 根据意图，将请求分发给 `handlers/` 目录中对应的处理器（如 `QueryHandler`）。
    3. 收集处理器的结果，并将其返回给上层应用。

### 2. `agents/` - 智能体集合

- **职责**: 包含一组专门化的LLM调用单元。每个智能体都针对一个特定的原子任务进行了优化，例如：
    - `IntentAgent`: 专注于理解用户输入的真实意图。
    - `ThinkingAgent`: 负责根据上下文、记忆和工具结果，生成深思熟虑的回答。
    - `RoutingAgent`: 决定处理用户请求的最佳路径（例如，是直接回答，还是使用工具）。
- **设计**: 每个智能体都通过 `PromptManager` 获取其定制的系统提示，确保其行为符合预期。

### 3. `handlers/` - 意图处理器

- **职责**: 实现特定意图的业务逻辑。每个 `Handler` 都继承自 `BaseHandler`。
- **示例**: `QueryHandler` 负责处理所有与“提问”相关的复杂逻辑，包括：
    1. 从 `hippocampus` 加载相关记忆。
    2. 使用 `RoutingAgent` 决定是否需要调用工具。
    3. 如果需要，则调用 `tools/` 中的工具。
    4. 调用 `ThinkingAgent` 生成最终答案。
    5. 更新 `hippocampus` 中的记忆。

### 4. `hippocampus/` - 记忆中心

- **职责**: 模拟生物海马体，管理AI的记忆。
- **分类**:
    - **长期记忆 (`memory_storage`)**: 存储跨会话的、持久化的信息和知识。
    - **短期会话记忆 (`session_storage`)**: 存储当前会话的上下文信息，如聊天记录。
- **机制**: 确保AI在对话中能够记住关键信息，提供连贯和个性化的交互体验。

### 5. `tools/` - 工具箱

- **职责**: 定义和管理AI可以使用的外部工具。
- **设计**:
    - `tool_registry.py`: 维护一个所有可用工具的注册表，供 `RoutingAgent` 和 `ThinkingAgent` 查询。
    - `native/`: 存放具体的工具实现，如文件操作、计算器等。
- **扩展性**: 可以轻松地通过添加新的工具文件来扩展AI的能力。

## 业务流程

一个典型的请求在 `cognitive_model` 内部的流转过程如下：

1.  **请求进入**: `Orchestrator` 接收到包含用户输入和会话ID的请求。
2.  **意图识别**: `Orchestrator` 调用 `IntentAgent`，判断用户意图为 "query"。
3.  **分发处理**: `Orchestrator` 将请求交给 `QueryHandler`。
4.  **记忆检索**: `QueryHandler` 从 `Hippocampus` 中检索与当前主题相关的记忆。
5.  **路径决策**: `QueryHandler` 调用 `RoutingAgent`，`RoutingAgent` 分析后认为需要使用文件分析工具来回答问题。
6.  **工具调用**: `QueryHandler` 从 `ToolRegistry` 获取 `file_analyzer` 工具并执行它。
7.  **思考与生成**: `QueryHandler` 将原始问题、检索到的记忆、工具执行结果一起传递给 `ThinkingAgent`。
8.  **答案形成**: `ThinkingAgent` 综合所有信息，生成最终的、结构化的答案。
9.  **记忆更新**: `QueryHandler` 调用 `SummarizationAgent` 对本次交互进行总结，并更新到 `Hippocampus` 中。
10. **返回结果**: `QueryHandler` 将最终答案返回给 `Orchestrator`，并最终回传给客户端。