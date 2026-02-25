# cognitive_model/handlers/base_handler.py

import abc
from typing import Dict, Any

class BaseHandler(abc.ABC):
    """
    所有意图处理器（Handler）的抽象基类（Abstract Base Class）。

    该基类定义了所有具体意图处理器必须遵循的统一接口规范。
    通过继承这个类，可以确保每种处理器都能被认知协调器（Orchestrator）以相同的方式调用，
    从而实现一个可扩展、可维护的模块化处理架构。

    核心设计思想:
    -   **统一接口 (Unified Interface)**: `BaseHandler` 的核心价值在于定义了一个通用的 `handle` 方法。
        这创建了一个强大的抽象层，使得 `Orchestrator` 无需关心每个意图的具体处理逻辑，
        只需知道如何调用 `handle` 方法即可。这种设计遵循了“依赖倒置原则”，
        高层模块（Orchestrator）不依赖于低层模块（具体Handler），而是依赖于抽象（BaseHandler）。
    -   **策略模式 (Strategy Pattern)**: 整个 Handler 架构是策略模式的一个典型应用。
        `BaseHandler` 定义了策略接口，而每个具体的 Handler（如 `QueryHandler`, `TuningHandler`）
        则是实现该接口的具体策略。`Orchestrator` 在运行时根据意图动态选择并执行相应的策略。
    -   **可扩展性 (Extensibility)**: 当需要支持新的用户意图时，开发者只需创建一个新的类，
        继承自 `BaseHandler` 并实现 `handle` 方法，然后在 `Orchestrator` 中注册这个新的 Handler 即可。
        这个过程不会影响任何现有的代码，符合“开闭原则”。
    -   **依赖注入 (Dependency Injection)**: `handle` 方法的参数设计体现了依赖注入的思想。
        它不是让 Handler 自己去获取所需的依赖（如 `Orchestrator`, `user_input` 等），
        而是在调用时由外部（`Orchestrator`）将这些依赖传递进来。这使得 Handler 更易于测试和解耦。

    """
    @abc.abstractmethod
    async def handle(self, orchestrator, user_input: str, history: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        处理用户输入和相关意图的核心抽象方法。

        每个具体的Handler都需要实现此方法，以定义针对特定意图（如"提问"、"自我介绍"等）的业务逻辑。

        Args:
            orchestrator: 认知协调器（Orchestrator）的实例。通过此实例，处理器可以访问模型的所有核心组件，
                          例如思考智能体（ThinkingAgent）、记忆海马体（Hippocampus）、工具集（Tools）等，
                          从而协同完成复杂的任务。
            user_input (str): 用户的原始输入字符串。
            session_id (str): 当前对话的会话ID，用于追踪和管理会话状态。
            session_tools (Dict[str, Any]): 本次请求专用的、动态加载的工具集。
            **kwargs: 一个包含额外参数的字典。这通常用于传递由前序模块（如IntentAgent）生成的信息，
                      其中最重要的就是 `intent`（意图），它是决定哪个Handler被调用的关键。

        Returns:
            Dict[str, Any]: 一个字典，其结构必须包含两个键：
                - "response" (Dict[str, Any]): 一个准备发送给客户端的结构化响应。该字典通常包含 `type`, `content` 等字段，
                                              用于前端展示和语音合成。
                - "full_assistant_response" (str): AI生成的完整、未经删减的原始文本回复。这个回复主要用于日志记录、
                                                     数据分析和模型调试，确保所有生成的内容都被完整保存。
                例如: {
                    "response": {"type": "answer", "content": "..."},
                    "full_assistant_response": "完整的思考过程和最终答案..."
                }
        """
        pass

    async def handle_stream(self, orchestrator, user_input: str, session_id: str, stream_callback=None, **kwargs) -> Dict[str, Any]:
        """
        处理用户输入和相关意图的流式版本方法。

        这是一个可选的方法，用于支持流式回复。如果Handler不需要特殊的流式处理逻辑，
        可以使用默认实现，它会调用普通的handle方法。

        Args:
            orchestrator: 认知协调器（Orchestrator）的实例。
            user_input (str): 用户的原始输入字符串。
            session_id (str): 当前对话的会话ID，用于追踪和管理会话状态。
            stream_callback (callable, optional): 流式回调函数，接收参数：
                - content (str): 当前接收到的内容片段
                - is_final (bool): 是否为最终完成状态
                - full_content (str): 到目前为止的完整内容
            **kwargs: 包含额外参数的字典。

        Returns:
            Dict[str, Any]: 与handle方法相同的返回格式。
        """
        # 默认实现：对于不需要特殊流式处理的Handler，直接调用普通handle方法
        return await self.handle(orchestrator, user_input, orchestrator.history, session_id, **kwargs)