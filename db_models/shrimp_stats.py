#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虾类统计数据模型
对应数据库表 shrimp_stats
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger
from sqlalchemy.dialects.mysql import DATETIME
from db_models.base import db


class ShrimpStats(db.Model):
    """虾类统计数据模型"""
    
    __tablename__ = 'shrimp_stats'
    
    # 主键
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # UUID标识
    uuid = Column(String(36), unique=True, nullable=True)
    
    # 池塘/摄像头标识
    pond_id = Column(String(255), nullable=False)
    
    # 输入子目录
    input_subdir = Column(String(255), nullable=False)
    
    # 输出目录
    output_dir = Column(String(512), nullable=False)
    
    # 源数据创建时间(ISO格式)
    created_at_source_iso = Column(String(64), nullable=False)
    
    # 源数据创建时间
    created_at_source = Column(DateTime, nullable=True)
    
    # 检测置信度
    conf = Column(Float, nullable=True)
    
    # IOU阈值
    iou = Column(Float, nullable=True)
    
    # 活虾数量
    total_live = Column(Integer, nullable=False)
    
    # 死虾数量
    total_dead = Column(Integer, nullable=False)
    
    # 尺寸统计
    size_min_cm = Column(Float, nullable=True)
    size_max_cm = Column(Float, nullable=True)
    size_mean_cm = Column(Float, nullable=True)
    size_median_cm = Column(Float, nullable=True)
    
    # 重量统计
    weight_min_g = Column(Float, nullable=True)
    weight_max_g = Column(Float, nullable=True)
    weight_mean_g = Column(Float, nullable=True)
    weight_median_g = Column(Float, nullable=True)
    
    # 源文件路径
    source_file = Column(String(512), nullable=True)
    
    # 创建时间
    created_at = Column(DateTime, default=datetime.now)
    
    # 更新时间
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'pond_id': self.pond_id,
            'input_subdir': self.input_subdir,
            'output_dir': self.output_dir,
            'created_at_source_iso': self.created_at_source_iso,
            'created_at_source': self.created_at_source.isoformat() if self.created_at_source else None,
            'conf': self.conf,
            'iou': self.iou,
            'total_live': self.total_live,
            'total_dead': self.total_dead,
            'size_min_cm': self.size_min_cm,
            'size_max_cm': self.size_max_cm,
            'size_mean_cm': self.size_mean_cm,
            'size_median_cm': self.size_median_cm,
            'weight_min_g': self.weight_min_g,
            'weight_max_g': self.weight_max_g,
            'weight_mean_g': self.weight_mean_g,
            'weight_median_g': self.weight_median_g,
            'source_file': self.source_file,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<ShrimpStats {self.id} - {self.pond_id} - Live: {self.total_live}, Dead: {self.total_dead}>'