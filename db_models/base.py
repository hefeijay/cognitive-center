# base.py

from dataclasses import asdict
from typing import Any, Dict

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass


class Base(MappedAsDataclass, DeclarativeBase):
    """
    SQLAlchemy基础模型类，支持数据类功能
    
    继承自MappedAsDataclass以支持dataclass参数如init=False
    这允许在mapped_column中使用dataclass相关的参数
    """
    def as_dict(self) -> Dict[str, Any]:
        """
        将模型实例转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含所有列名和值的字典
        """
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


db = SQLAlchemy(model_class=Base)



