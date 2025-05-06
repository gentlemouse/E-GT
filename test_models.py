import time
from colorama import Fore, Style
from models.model import OpenAIModel
from models.model import DeepSeekModel
from models.model import GeminiModel
from models.model import KimiModel
from models.model import ZhipuModel
from models.model import DoubaoModel
from models.model import XunfeiModel
from models.model import HunyuanModel
from models.model import WenxinModel
import config


def test_model(model, model_name: str, test_prompt: str = "你好") -> dict:
    """测试单个模型的API连通性"""
    result = {
        "model": model_name,
        "status": "未知",
        "response": None,
        "latency": None,
        "error": None
    }

    try:
        start_time = time.time()
        response = model.generate_response(test_prompt)
        latency = time.time() - start_time

        result.update({
            "status": "成功" if response else "无响应",
            "response": response[:50] + "..." if response else None,
            "latency": f"{latency:.2f}s",
            "error": None
        })

    except Exception as e:
        result.update({
            "status": "失败",
            "error": str(e)
        })

    return result


def print_test_result(result: dict):
    """可视化打印测试结果"""
    color = Fore.GREEN if result["status"] == "成功" else Fore.RED
    status_icon = "✓" if result["status"] == "成功" else "✗"

    print(f"\n{color}{status_icon} {result['model']}测试结果{Style.RESET_ALL}")
    print(f"状态: {color}{result['status']}{Style.RESET_ALL}")

    if result["status"] == "成功":
        print(f"响应片段: {result['response']}")
        print(f"延迟: {result['latency']}")
    else:
        print(f"错误信息: {Fore.YELLOW}{result['error']}{Style.RESET_ALL}")


def main():
    # 初始化测试模型列表（星火需补充api_secret）
    models = [
        (DeepSeekModel(config.DEEPSEEK_KEY), "deepseek"),
        (GeminiModel(config.GEMINI_KEY), "gemini-1.5-pro"),
        (KimiModel(config.KIMI_KEY), "moonshot-v1-32k"),
        (ZhipuModel(config.ZHIPU_KEY), "glm-4-flash"),
        (DoubaoModel(config.DOUBAO_KEY), "doubao-1-5-pro-32k-250115"),
        (XunfeiModel(config.XUNFEI_KEY), "4.0 Ultra"),
        (HunyuanModel(config.HUNYUAN_KEY), "hunyuan-turbo"),
        (WenxinModel(config.BAIDU_KEY, "WGWaPA7PS0OqY3WvawLTGgN1PjfTNfDw"), "ERNIE-4.0-Turbo-8K-latest")
    ]
    test_models = [
        (GeminiModel(config.GEMINI_KEY), "gemini-1.5-pro")
    ]

    print(f"{Fore.CYAN}开始API连通性测试...{Style.RESET_ALL}")

    for model, name in test_models:
        result = test_model(model, name)
        print_test_result(result)
        time.sleep(1)  # 避免频繁调用


if __name__ == "__main__":
    main()