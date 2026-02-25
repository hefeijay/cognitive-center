"""
数据库模型：用于存储和管理所有AI智能体的提示（Prompt）模板。
"""

from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, text, UniqueConstraint
from .base import Base

class Prompt(Base):
    """
    Prompt模型，代表一个独立的提示模板。

    这个模型的设计旨在将 `prompts.json` 文件的结构化数据映射到数据库表中。
    每个 agent 的每个 prompt 模板都将作为一条记录存储。

    Attributes:
        id (int): 主键ID。
        agent_name (str): 智能体的名称，用于唯一标识一个智能体（例如 'intent_agent'）。
        template_key (str, optional): 模板的键。对于拥有多个模板的智能体（如 'thinking_agent'），
                                     此字段用于区分不同的模板（如 'with_tool', 'without_tool'）。
                                     对于只有一个主模板的智能体，此字段为 NULL。
        description (str, optional): 对该智能体或模板用途的简要描述。
        version (str, optional): 模板的版本号。
        template (str): 完整的提示模板字符串，其中可能包含格式化占位符（例如 `{user_input}`）。
        created_at (datetime): 记录创建时间。
        updated_at (datetime): 记录更新时间。
    """
    __tablename__ = 'prompts'

    id = Column(Integer, primary_key=True, index=True, comment="主键ID")
    agent_name = Column(String(255), nullable=False, index=True, comment="智能体名称")
    template_key = Column(String(255), nullable=True, index=True, comment="模板键，用于区分同一智能体的不同模板")
    
    description = Column(Text, nullable=True, comment="模板描述")
    version = Column(String(50), nullable=True, comment="模板版本")
    template = Column(Text, nullable=False, comment="提示模板内容")
    
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), comment="创建时间")
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"), comment="更新时间")

    __table_args__ = (
        UniqueConstraint('agent_name', 'template_key', name='uq_agent_name_template_key'),
    )

    def __repr__(self):
        return f"<Prompt(agent_name='{self.agent_name}', template_key='{self.template_key}')>"