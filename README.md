# Cognitive Center

`cognitive-center` 是日本陆上养殖相关系统中的认知决策与多智能体处理项目。

## 当前有效启动方式

当前实际使用时，优先以 `cognitive_graph/main.py` 为入口。

持续运行模式命令：

```bash
cd /home/gmm/srv/cognitive-center
python -u cognitive_graph/main.py --mode continuous
```

补充说明：

- 这条命令是从 `cognitive-center` 项目根目录执行的写法。
- `cognitive_graph/main.py` 当前支持 `interactive`、`continuous`、`status`、`test` 四种模式。
- 启动前需要先安装 `cognitive_graph/requirements.txt` 中的依赖，并准备好项目根目录下的 `.env`。

## 目录说明

- `cognitive_graph/`：当前实际运行入口，负责模式解析、持续处理和状态查看。
- `cognitive_model/`：认知编排子系统，当前这份 README 不展开其内部模块细节。
- `db_models/`：本项目自己的数据库模型定义。

## 当前建议理解

1. 如果目的是排查当前运行中的认知处理链，优先看 `cognitive_graph/`。
2. 当前持续运行命令应以 `python -u cognitive_graph/main.py --mode continuous` 为准。
3. 本项目与 `japan-aquaculture-project/backend` 存在数据库模型和路径层面的耦合，换机或重构前要先确认依赖边界。

## 常用入口

- `cognitive_graph/main.py`
- `cognitive_graph/config.py`
- `cognitive_graph/agents.py`

## 已知注意事项

- 当前代码里存在写死路径。
- `db_models` 在本项目和 `japan-aquaculture-project/backend` 中都存在，容易产生漂移。
- 旧文档中部分文件名、配置名和目录结构与仓库现状不完全一致。
- `cognitive_graph/main.py` 当前会追加写死的 Python 路径，并依赖项目根目录下的 `.env` 与外部数据库/LLM 配置。
