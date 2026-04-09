# `cognitive_model`

`cognitive_model` 是 `cognitive-center` 中一套较底层的认知编排组件，包含 Orchestrator、Agent、Handler、记忆模块和工具注册逻辑。

但对当前实际运行命令：

```bash
python -u cognitive_graph/main.py --mode continuous
```

需要先明确一件事：这条启动链路当前优先走的是 `cognitive_graph`，不是本目录。

## 当前启动链路

对 `continuous` 模式，建议优先按下面顺序理解代码：

1. `../cognitive_graph/main.py`
2. `../cognitive_graph/agents.py`
3. `../cognitive_graph/config.py`
4. `../cognitive_graph/database.py` 以及相关 `db_models/`

当前这条链路里，`main.py` 直接实例化的是 `cognitive_graph.agents.MultiAgentCollaborationFramework`，持续处理逻辑也在 `cognitive_graph/agents.py` 内部完成。

这意味着：

- `cognitive_model/handlers/query_handler.py` 不是这条 `continuous` 启动路径的直接依赖
- 本目录里即便存在历史遗留代码或旧导入，也不会自动影响这条命令
- 排查 `continuous` 模式问题时，不要先从 `cognitive_model` 入手

## 本目录的定位

`cognitive_model` 更适合看作：

- 一套认知能力组件库
- 另一套编排思路的实现
- 扩展 Agent / Handler / Memory / Tool 逻辑时可复用的基础模块

而不是当前 `continuous` 模式的主入口。

## 什么时候需要看这里

只有在下面这些场景，才建议回到 `cognitive_model` 深挖：

- 需要调整 `orchestrator.py` 这套认知编排逻辑
- 需要修改 `handlers/` 下的意图处理策略
- 需要扩展 `agents/`、`tools/`、`hippocampus/` 的能力
- 上层运行链路未来明确切换回 `cognitive_model` 编排器

如果只是维护当前在线运行命令，优先看 `cognitive_graph` 即可。

## 目录概览

- `orchestrator.py`：认知协调器入口
- `agents/`：意图识别、路由、思考、总结等 Agent
- `handlers/`：不同意图的处理逻辑
- `hippocampus/`：会话与记忆相关能力
- `config/`：提示词与配置管理
- `tools/`：工具注册与调用封装
- `tasks/`：任务处理相关逻辑

## 维护说明

- 旧文档里如果把 `cognitive_model` 写成唯一入口，应以当前代码为准修正理解
- 旧说明里出现的部分结构或外部依赖，可能只是历史背景，不应直接当作当前 `continuous` 模式的必需项
- 如果后续确认运行链路发生切换，再同步更新本 README，避免把“能力模块”和“真实入口”混在一起