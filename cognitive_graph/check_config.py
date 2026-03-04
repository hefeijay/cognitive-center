#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置检查脚本
验证 OpenRouter 和数据库配置是否正确
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
env_path = os.path.join(os.path.dirname(__file__), '.env')
if not os.path.exists(env_path):
    print("❌ 错误: .env 文件不存在")
    print("💡 提示: 请运行 'cp .env.example .env' 并配置相关参数")
    sys.exit(1)

load_dotenv(env_path)

# 添加项目路径
sys.path.append('/home/gmm/srv/cognitive-center')
# 使用 japan-aquaculture-project 的数据库模型
sys.path.append('/home/gmm/srv/japan-aquaculture-project/backend')

print("🔍 开始检查配置...\n")
print("=" * 60)

# 检查必需的环境变量
errors = []
warnings = []

# 1. 检查 OpenRouter API Key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    errors.append("OPENROUTER_API_KEY 未设置")
elif not api_key.startswith("sk-or-v1-"):
    warnings.append(f"OPENROUTER_API_KEY 格式可能不正确 (应以 'sk-or-v1-' 开头)")
else:
    print(f"✓ OPENROUTER_API_KEY: {api_key[:20]}...")

# 2. 检查 OpenRouter Base URL
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
print(f"✓ OPENROUTER_BASE_URL: {base_url}")

# 3. 检查模型配置
model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
print(f"✓ OPENROUTER_MODEL: {model}")

print()

# 4. 检查数据库配置
db_url = os.getenv("DATABASE_URL")
if not db_url:
    errors.append("DATABASE_URL 未设置")
else:
    print(f"✓ DATABASE_URL: {db_url}")
    
    # 如果是 SQLite，检查文件是否存在
    if db_url.startswith("sqlite"):
        db_file = db_url.replace("sqlite:///", "").replace("./", "")
        if os.path.exists(db_file):
            print(f"✓ 数据库文件存在: {db_file}")
        else:
            warnings.append(f"数据库文件不存在: {db_file} (首次运行会自动创建)")

print()
print("=" * 60)

# 显示错误和警告
if errors:
    print("\n❌ 配置错误:")
    for error in errors:
        print(f"  - {error}")

if warnings:
    print("\n⚠️  警告:")
    for warning in warnings:
        print(f"  - {warning}")

if not errors and not warnings:
    print("\n✅ 所有配置检查通过!")

# 测试导入配置模块
if not errors:
    print("\n🔧 测试导入配置模块...")
    try:
        from cognitive_graph.config import config
        print(f"✓ 配置模块加载成功")
        print(f"✓ API Key 已设置: {config.OPENROUTER_API_KEY[:20]}...")
        print(f"✓ 模型: {config.OPENROUTER_MODEL}")
        print(f"✓ Base URL: {config.OPENROUTER_BASE_URL}")
        print(f"✓ 数据库: {config.DATABASE_URL}")
    except Exception as e:
        print(f"❌ 配置模块加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 测试 OpenRouter 连接
if not errors:
    print("\n🌐 测试 OpenRouter 连接...")
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            timeout=10
        )
        
        # 发送一个简单的测试请求
        response = llm.invoke("Hello")
        print(f"✓ OpenRouter 连接成功!")
        print(f"✓ 测试响应: {response.content[:100]}...")
        
    except Exception as e:
        print(f"❌ OpenRouter 连接失败: {e}")
        print("\n💡 可能的原因:")
        print("  1. API Key 无效或已过期")
        print("  2. 网络连接问题")
        print("  3. OpenRouter 服务暂时不可用")
        print("  4. 模型名称不正确或无权访问")
        sys.exit(1)

print("\n" + "=" * 60)
if not errors:
    print("🎉 配置检查完成，系统已就绪!")
    print("\n运行方式:")
    print("  - 快速启动: ./run.sh")
    print("  - 持续模式: python main.py --mode continuous")
    print("  - 交互模式: python main.py --mode interactive")
    print("  - 查看帮助: python main.py --help")
else:
    print("❌ 请修复上述错误后重试")
    sys.exit(1)
