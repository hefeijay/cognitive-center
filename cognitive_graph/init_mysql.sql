-- ===================================
-- MySQL 数据库初始化脚本
-- ===================================
-- 创建数据库 cognitive
-- 字符集使用 utf8mb4 以支持 emoji 和多语言

-- 1. 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS cognitive
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- 2. 使用数据库
USE cognitive;

-- 3. 显示数据库信息
SELECT 
    '数据库创建成功！' AS message,
    DATABASE() AS current_database,
    @@character_set_database AS charset,
    @@collation_database AS collation;

-- ===================================
-- 说明
-- ===================================
-- 运行此脚本后，请执行以下命令创建数据表：
-- python create_tables.py
--
-- 这将自动创建以下表：
--   - message_queue: 消息队列表
--   - ai_decisions: AI决策表
--   - chat_history: 聊天历史表
--   - prompts: 智能体提示词表
-- ===================================
