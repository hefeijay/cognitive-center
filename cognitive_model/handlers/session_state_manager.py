# cognitive_model/handlers/session_state_manager.py

from typing import Dict, Any

class SessionStateManager:
    """
    管理与每个独立会话（session）相关的临时状态信息。

    在认知模型中，某些操作（尤其是异步工具调用）需要在不同的时间点之间传递状态。
    例如，当一个工具开始异步执行后，需要存储一个标记，以便在后续的轮询中检查其状态。
    这个类提供了一个简单的、基于内存的键值存储，以会话ID为作用域，来处理这类需求。

    主要功能:
    - 为每个会话维护一个独立的、隔离的状态字典。
    - 提供设置、获取和清理特定状态的原子操作。
    """
    def __init__(self):
        """
        初始化会话状态管理器。
        `session_states` 是一个字典，其键是会话ID（session_id），值是另一个字典，
        该内部字典存储了该会话的所有键值对状态。
        """
        self.session_states: Dict[str, Dict[str, Any]] = {}

    def set_state(self, session_id: str, key: str, value: Any):
        """
        为一个指定的会话设置一个键值对状态。

        如果会话ID首次出现，会自动为该会话创建一个新的状态字典。

        Args:
            session_id (str): 会话的唯一标识符。
            key (str): 要设置的状态的键。
            value (Any): 要设置的状态的值。
        """
        if session_id not in self.session_states:
            self.session_states[session_id] = {}
        self.session_states[session_id][key] = value

    def get_state(self, session_id: str, key: str, default: Any = None) -> Any:
        """
        从指定的会话中获取一个状态的值。

        如果会话ID或键不存在，将返回指定的默认值，以避免 `KeyError`。

        Args:
            session_id (str): 会话的唯一标识符。
            key (str): 要获取的状态的键。
            default (Any, optional): 如果键不存在时返回的默认值。默认为 `None`。

        Returns:
            Any: 存储的状态值，如果不存在则为默认值。
        """
        return self.session_states.get(session_id, {}).get(key, default)

    def clear_state(self, session_id: str, key: str):
        """
        从指定的会话中清理（删除）一个特定的状态。

        这在某个临时状态（如异步任务标记）完成后非常有用，可以防止状态的无限累积。

        Args:
            session_id (str): 会话的唯一标识符。
            key (str): 要清理的状态的键。
        """
        if session_id in self.session_states and key in self.session_states[session_id]:
            del self.session_states[session_id][key]