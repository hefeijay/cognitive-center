# 多智能体协作系统

基于LangGraph的多智能体协作持续多轮对话框架，支持从数据库读取消息、智能体协作处理、输出决策建议等功能。

## 功能特性

- 🤖 **多智能体协作**: 基于LangGraph实现的多智能体协作框架
- 💬 **持续多轮对话**: 支持基于对话历史的多轮对话处理
- 🗄️ **数据库集成**: 完整的数据库操作，包括消息队列、对话历史、决策存储
- ⚙️ **配置化Prompt**: 智能体Prompt支持数据库配置管理
- 🔧 **MCP服务集成**: 支持调用第三方MCP服务作为智能体工具
- 📊 **决策输出**: 自动生成并存储结构化的决策建议

## 项目结构

```
cognitive_graph/
├── main.py                 # 主程序入口
├── config.py              # 配置管理
├── database.py            # 数据库操作
├── agents.py              # 多智能体框架
├── mcp_tools.py           # MCP服务集成
├── test_framework.py      # 测试框架
└── README.md             # 项目说明
```

## 数据库模型

项目使用以下数据库表：

- `message_queue`: 消息队列表，存储待处理的消息
- `ai_decision`: AI决策表，存储智能体生成的决策建议
- `chat_history`: 对话历史表，存储多轮对话记录
- `prompt`: Prompt配置表，存储智能体的提示词配置

## 安装依赖

```bash
cd /usr/henry/cognitive-center/cognitive_graph
pip install -r requirements.txt
```

## 配置环境

在 `/usr/henry/cognitive-center/.env` 文件中配置以下环境变量：

```env
DATABASE_URL=sqlite:///cognitive_center.db
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用方法

### 1. 交互模式

```bash
python main.py --mode interactive
```

在交互模式下，您可以直接输入消息，系统会进行多智能体协作分析并返回决策建议。

### 2. 持续处理模式

```bash
python main.py --mode continuous
```

持续处理模式会监控数据库中的消息队列，自动处理新的消息。

### 3. 测试模式

```bash
# 使用默认测试消息
python main.py --mode test

# 使用自定义测试消息
python main.py --mode test --message "传感器检测到温度异常"
```

### 4. 查看系统状态

```bash
python main.py --mode status
```

### 5. 运行综合测试

```bash
python test_framework.py
```

## 智能体配置

系统支持通过数据库配置智能体的Prompt。在 `prompt` 表中添加记录：

```sql
INSERT INTO prompt (template_key, agent_name, content, variables, status) VALUES 
('graph_agent', 'researcher', '你是一个专业的研究员...', '{"role": "researcher"}', 'active'),
('graph_agent', 'writer', '你是一个专业的决策建议撰写员...', '{"role": "writer"}', 'active');
```

## MCP服务集成

系统支持集成第三方MCP服务作为智能体工具。在配置文件中设置MCP服务URL：

```python
MCP_SERVICE_URL = "http://localhost:8080/mcp"
```

## API接口

### 创建消息

```python
from cognitive_graph.database import DatabaseManager

with DatabaseManager() as db:
    message = db.create_message(
        message_id="msg_001",
        content="用户输入的消息内容",
        message_type="user_input"
    )
```

### 处理消息

```python
from cognitive_graph.agents import MultiAgentCollaborationFramework

framework = MultiAgentCollaborationFramework()
result = framework.process_message("msg_001")
```

## 配置参数

主要配置参数说明：

- `MAX_CONVERSATION_ROUNDS`: 最大对话轮次 (默认: 5)
- `MESSAGE_POLL_INTERVAL`: 消息轮询间隔秒数 (默认: 5)
- `AGENT_TEMPERATURE`: 智能体温度参数 (默认: 0.7)
- `MAX_TOKENS`: 最大token数 (默认: 1000)

## 监控和日志

系统会自动记录：

- 消息处理状态和时间
- 智能体对话历史
- 决策生成结果
- 错误和异常信息

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查 `.env` 文件中的 `DATABASE_URL` 配置
   - 确保数据库文件权限正确

2. **OpenAI API调用失败**
   - 检查 `OPENAI_API_KEY` 是否正确配置
   - 确认API密钥有效且有足够额度

3. **MCP服务连接失败**
   - 检查MCP服务是否正常运行
   - 确认MCP服务URL配置正确

### 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 开发指南

### 添加新的智能体

1. 在数据库中添加新的Prompt配置
2. 在 `agents.py` 中的 `_create_agent_node` 方法中添加新智能体逻辑
3. 更新路由逻辑以包含新智能体

### 扩展MCP工具

1. 在 `mcp_tools.py` 中添加新的工具类
2. 在 `MCPServiceManager` 中注册新工具
3. 更新智能体配置以使用新工具

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。