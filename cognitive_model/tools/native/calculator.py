# cognitive_model/tools/native/calculator.py

import logging
import numexpr

# 初始化日志记录器，用于记录计算器工具的使用情况
logger = logging.getLogger("app.cognitive_model.tools.calculator")

def calculate(expression: str) -> str:
    """
    安全地计算一个数学字符串表达式。

    核心功能:
    -   使用 `numexpr` 库代替 `eval()` 来执行数学计算，从而避免 `eval()` 可能带来的严重安全风险（如执行任意代码）。
    -   支持包含变量、复杂运算和函数的表达式。

    业务逻辑:
    -   接收一个字符串形式的数学表达式。
    -   记录传入的表达式以供调试。
    -   调用 `numexpr.evaluate()` 执行计算。
    -   将计算结果（通常是 NumPy 类型）转换为 Python 的原生数值类型，然后格式化为字符串返回。
    -   如果计算过程中发生任何异常（如语法错误、无效操作），则捕获异常，记录错误日志，并返回一个格式化的错误信息字符串。

    :param expression: 要计算的数学表达式，例如 "12 * 34" 或 "(2 + 3) * 5"。
    :return: 计算结果的字符串形式，或在出错时返回一个描述错误的字符串。
    """
    logger.info(f"正在使用计算器工具，计算表达式: '{expression}'")
    try:
        # `numexpr.evaluate()` 是一个专门为数值计算设计的安全、高性能的表达式求值器。
        # 它可以处理复杂的数学运算，但不会执行任意的 Python 代码，因此是安全的。
        result = numexpr.evaluate(expression)
        
        # `result.item()` 将 NumPy 数组中的单个元素转换为等效的 Python 标量（如 int, float）。
        output = str(result.item())
        
        logger.info(f"计算结果: {output}")
        return output
    except Exception as e:
        # 捕获所有可能的计算错误，例如语法错误、未定义的变量等。
        logger.error(f"计算表达式 '{expression}' 时出错: {e}")
        return f"计算错误: {str(e)}"