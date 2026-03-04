#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体协作系统主程序
基于LangGraph的持续多轮对话框架

功能特性:
1. 持续处理模式 - 监控消息队列并自动处理
2. 交互模式 - 用户输入消息进行实时处理
3. 测试模式 - 批量测试消息处理功能
4. 状态监控 - 显示系统运行状态和统计信息
5. 数据库管理 - 消息队列和决策记录管理
"""

import sys
import uuid
import json
import argparse
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append('/home/gmm/srv/cognitive-center')
# 使用 japan-aquaculture-project 的数据库模型
sys.path.append('/home/gmm/srv/japan-aquaculture-project/backend')

from cognitive_graph.agents import MultiAgentCollaborationFramework
from cognitive_graph.database import DatabaseManager
from cognitive_graph.config import config


def create_test_message(content: str, message_type: str = "user_input", priority: int = 5) -> str:
    """
    创建测试消息
    
    Args:
        content: 消息内容
        message_type: 消息类型，默认为user_input
        priority: 消息优先级，1-10，数字越小优先级越高
        
    Returns:
        str: 消息ID，失败时返回None
    """
    message_id = f"msg_{uuid.uuid4()}"
    
    try:
        with DatabaseManager() as db:
            message = db.create_message(
                message_id=message_id,
                content=content,
                message_type=message_type,
                priority=priority,
                metadata=json.dumps({
                    "source": "main.py",
                    "created_by": "system",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0"
                })
            )
            print(f"✓ 创建消息: {message_id}")
            print(f"  内容: {content[:100]}{'...' if len(content) > 100 else ''}")
            return message_id
    except Exception as e:
        print(f"✗ 创建消息失败: {e}")
        return None


def process_single_message(content: str, show_details: bool = True) -> Optional[str]:
    """
    处理单个消息并返回结果
    
    Args:
        content: 消息内容
        show_details: 是否显示详细处理信息
        
    Returns:
        str: 处理结果，失败时返回None
    """
    if show_details:
        print(f"\n{'='*60}")
        print(f"开始处理消息: {content}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # 创建测试消息
    message_id = create_test_message(content)
    if not message_id:
        return None
    
    # 初始化多智能体框架
    try:
        framework = MultiAgentCollaborationFramework()
        if show_details:
            print("✓ 多智能体框架初始化成功")
    except Exception as e:
        print(f"✗ 多智能体框架初始化失败: {e}")
        return None
    
    # 处理消息
    try:
        result = framework.process_message(message_id)
        processing_time = time.time() - start_time
        
        if show_details:
            print(f"✓ 消息处理完成")
            print(f"  处理时间: {processing_time:.2f}秒")
            print(f"  消息ID: {message_id}")
            if result:
                print(f"  处理结果: {result}")
            else:
                print(f"  处理结果: 无")
            
        return result
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"✗ 处理消息失败: {e}")
        print(f"  处理时间: {processing_time:.2f}秒")
        return None


def run_continuous_mode():
    """
    运行持续处理模式
    监控消息队列并自动处理新消息
    """
    print("🚀 启动多智能体协作系统 - 持续处理模式")
    print(f"📊 系统配置:")
    print(f"  数据库: {config.DATABASE_URL.split('/')[-1] if '/' in config.DATABASE_URL else config.DATABASE_URL}")
    print(f"  OpenRouter模型: {config.OPENROUTER_MODEL}")
    print(f"  最大对话轮次: {config.MAX_CONVERSATION_ROUNDS}")
    print(f"  消息轮询间隔: {config.MESSAGE_POLL_INTERVAL}秒")
    print(f"  消息批处理大小: {config.MESSAGE_BATCH_SIZE}")
    print("-" * 60)
    
    # 初始化多智能体框架
    try:
        framework = MultiAgentCollaborationFramework()
        print("✓ 多智能体框架初始化成功")
    except Exception as e:
        print(f"✗ 多智能体框架初始化失败: {e}")
        return
    
    # 开始持续处理
    try:
        print("🔄 开始监控消息队列...")
        framework.run_continuous_processing()
    except KeyboardInterrupt:
        print("\n⏹️  收到中断信号，正在停止系统...")
    except Exception as e:
        print(f"✗ 持续处理模式出错: {e}")


def run_interactive_mode():
    """
    运行交互模式
    用户可以输入消息进行实时处理
    """
    print("💬 启动多智能体协作系统 - 交互模式")
    print("📝 输入消息内容，系统将进行多智能体协作分析")
    print("💡 支持的命令:")
    print("   - 输入任意文本进行分析")
    print("   - 'status' 查看系统状态")
    print("   - 'help' 显示帮助信息")
    print("   - 'quit' 或 'exit' 退出")
    print("-" * 60)
    
    command_count = 0
    
    while True:
        try:
            user_input = input(f"\n[{command_count + 1}] 请输入消息: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 退出交互模式")
                break
            elif user_input.lower() == 'status':
                show_system_status()
                continue
            elif user_input.lower() == 'help':
                print("\n📖 帮助信息:")
                print("  - 输入任意文本: 系统将使用多智能体协作进行分析")
                print("  - status: 显示系统当前状态")
                print("  - help: 显示此帮助信息")
                print("  - quit/exit: 退出交互模式")
                continue
            
            if not user_input:
                print("⚠️  请输入有效的消息内容")
                continue
            
            command_count += 1
            result = process_single_message(user_input, show_details=True)
            
            if result:
                print(f"\n🎯 决策结果:")
                print("-" * 40)
                print(result)
                print("-" * 40)
            else:
                print("❌ 处理失败，请检查系统配置和网络连接")
                
        except KeyboardInterrupt:
            print("\n⏹️  收到中断信号，退出交互模式")
            break
        except Exception as e:
            print(f"❌ 交互模式出错: {e}")


def show_system_status():
    """
    显示系统状态
    包括消息队列、AI决策、对话历史等统计信息
    """
    print("\n📊 多智能体协作系统状态")
    print("=" * 60)
    
    try:
        with DatabaseManager() as db:
            # 检查待处理消息
            pending_messages = db.get_pending_messages(limit=10)
            processing_messages = db.get_processing_messages(limit=5)
            
            print(f"📬 消息队列状态:")
            print(f"  待处理消息: {len(pending_messages)}")
            print(f"  处理中消息: {len(processing_messages)}")
            
            if pending_messages:
                print(f"\n📋 最近的待处理消息:")
                for i, msg in enumerate(pending_messages[:5], 1):
                    print(f"  {i}. [{msg.priority}] {msg.content[:60]}{'...' if len(msg.content) > 60 else ''}")
            
            # 检查最近的AI决策
            from db_models.ai_decision import AIDecision
            recent_decisions = (db.session.query(AIDecision)
                              .filter(AIDecision.status == 'active')
                              .order_by(AIDecision.created_at.desc())
                              .limit(5)
                              .all())
            
            print(f"\n🧠 AI决策状态:")
            print(f"  最近决策数量: {len(recent_decisions)}")
            
            if recent_decisions:
                print(f"\n📝 最近的AI决策:")
                for i, decision in enumerate(recent_decisions, 1):
                    print(f"  {i}. [{decision.confidence:.1f}%] {decision.message[:60]}{'...' if len(decision.message) > 60 else ''}")
            
            # 检查对话历史
            from db_models.chat_history import ChatHistory
            recent_chats = (db.session.query(ChatHistory)
                           .order_by(ChatHistory.timestamp.desc())
                           .limit(10)
                           .all())
            
            print(f"\n💬 对话历史:")
            print(f"  最近对话数量: {len(recent_chats)}")
            
            # 系统配置信息
            print(f"\n⚙️  系统配置:")
            print(f"  数据库: {config.DATABASE_URL.split('/')[-1] if '/' in config.DATABASE_URL else 'SQLite'}")
            print(f"  最大对话轮次: {config.MAX_CONVERSATION_ROUNDS}")
            print(f"  消息轮询间隔: {config.MESSAGE_POLL_INTERVAL}秒")
            print(f"  消息批处理大小: {config.MESSAGE_BATCH_SIZE}")
            
    except Exception as e:
        print(f"❌ 获取系统状态失败: {e}")


def run_batch_test(test_messages: List[str] = None):
    """
    运行批量测试
    
    Args:
        test_messages: 测试消息列表，如果为None则使用默认测试消息
    """
    if test_messages is None:
        test_messages = [
            "传感器检测到温度异常，当前温度35°C，需要立即处理",
            "用户反馈系统响应缓慢，影响正常使用体验",
            "检测到网络连接不稳定，可能影响数据传输",
            "设备电量低于20%，建议及时充电",
            "发现潜在安全漏洞，需要紧急修复",
            "系统内存使用率达到90%，建议清理缓存",
            "数据库连接超时，请检查网络状态",
            "用户登录失败次数过多，可能存在安全风险"
        ]
    
    print(f"🧪 开始批量测试 - 共{len(test_messages)}条消息")
    print("=" * 60)
    
    results = []
    success_count = 0
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[{i}/{len(test_messages)}] 测试消息: {message[:80]}{'...' if len(message) > 80 else ''}")
        
        start_time = time.time()
        result = process_single_message(message, show_details=True)
        processing_time = time.time() - start_time
        
        if result:
            success_count += 1
            status = "✓ 成功"
            print(f"  {status} - 处理时间: {processing_time:.2f}秒")
        else:
            status = "✗ 失败"
            print(f"  {status} - 处理时间: {processing_time:.2f}秒")
        
        results.append({
            "message": message,
            "success": result is not None,
            "result": result,
            "processing_time": processing_time
        })
    
    # 显示测试总结
    print(f"\n📈 批量测试总结:")
    print("=" * 60)
    print(f"总测试数: {len(test_messages)}")
    print(f"成功数: {success_count}")
    print(f"失败数: {len(test_messages) - success_count}")
    print(f"成功率: {success_count / len(test_messages) * 100:.1f}%")
    
    if results:
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        print(f"平均处理时间: {avg_time:.2f}秒")
    
    return results


def main():
    """
    主函数
    解析命令行参数并启动相应的运行模式
    """
    parser = argparse.ArgumentParser(
        description="多智能体协作系统 - 基于LangGraph的智能决策框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明:
  continuous  - 持续处理模式，监控消息队列并自动处理
  interactive - 交互模式，用户输入消息进行实时处理  
  status      - 显示系统状态和统计信息
  test        - 测试模式，批量测试消息处理功能

使用示例:
  python main.py --mode interactive
  python main.py --mode test --message "测试消息内容"
  python main.py --mode continuous
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["continuous", "interactive", "status", "test"],
        default="interactive",
        help="运行模式 (默认: interactive)"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="测试模式下的单个消息内容"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="批量测试时的消息数量 (默认: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出信息"
    )
    
    args = parser.parse_args()
    
    # 显示启动信息
    print("🤖 多智能体协作系统")
    print(f"⚡ 运行模式: {args.mode}")
    print(f"🕒 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.mode == "continuous":
            run_continuous_mode()
        elif args.mode == "interactive":
            run_interactive_mode()
        elif args.mode == "status":
            show_system_status()
        elif args.mode == "test":
            if args.message:
                print(f"\n🧪 测试单个消息: {args.message}")
                result = process_single_message(args.message)
                if result:
                    print(f"\n🎯 处理结果:")
                    print("-" * 40)
                    print(result)
                    print("-" * 40)
                else:
                    print("❌ 处理失败")
            else:
                # 批量测试
                run_batch_test()
                
    except KeyboardInterrupt:
        print("\n⏹️  程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        print(f"\n👋 程序结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()