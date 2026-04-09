# `cognitive_graph`

`cognitive_graph` 是 `cognitive-center` 里当前可直接运行的多智能体处理链，支持交互模式、持续处理模式、状态查看和测试模式。

## 当前已确认入口

- 启动文件：`main.py`
- 模式：`interactive`、`continuous`、`status`、`test`
- 配置文件：`config.py`

## 安装与启动

```bash
cd /home/gmm/srv/cognitive-center/cognitive_graph
pip install -r requirements.txt
python main.py --mode interactive
```

其他模式：

```bash
python main.py --mode continuous
python main.py --mode status
python main.py --mode test
```

## 环境变量

- 数据库：`DATABASE_URL` 或 `MYSQL_*`
- LLM：`OPENROUTER_API_KEY`、`OPENROUTER_MODEL`、`OPENROUTER_BASE_URL`
- 路径：`PROJECT_ROOT_PATH`、`COGNITIVE_MODEL_PATH`
- MCP：`MCP_SERVER_URL`

注意：

1. `OPENROUTER_API_KEY` 是当前运行的硬性依赖。
2. 默认数据库是 `sqlite:///cognitive_graph.db`。
3. 只有当 `DATABASE_URL` 保持默认值且完整提供了 `MYSQL_*` 时，代码才会自动拼成 MySQL 连接。

## 工作目录要求

`config.py` 中 `.env` 的读取是相对当前工作目录的，因此建议在 `cognitive_graph/` 目录下运行。  
有些旧文档写 `python cognitive_graph/main.py ...`，那是从上一级目录执行时的写法，不能和当前目录启动方式混用。

## 目录说明

- `main.py`：入口和模式解析
- `agents.py`：多智能体协作框架
- `database.py`：数据库访问
- `config.py`：配置与 OpenRouter 初始化
- `mcp_tools.py`：MCP 相关模块
- `test/`：一些测试脚本

## 已知注意事项

1. 旧 README 中提到的 `test_framework.py` 并不存在。
2. 旧 README 使用 `OPENAI_API_KEY`、`MCP_SERVICE_URL`、`cognitive_center.db` 等配置名或默认值，和当前代码不一致。
3. 当前代码里虽然有 `mcp_tools.py`，但主图运行链路并没有直接把 MCP 工具接进来，不能把它写成默认已启用能力。
4. 代码里存在写死路径 `/home/gmm/srv/cognitive-center` 和 `/home/gmm/srv/japan-aquaculture-project/backend`，换机部署时必须额外检查。

## 建议优先阅读

- `main.py`
- `config.py`
- `agents.py`
- `database.py`
- `.env.example`