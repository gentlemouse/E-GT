from models.model import OpenAIModel
from models.model import DeepSeekModel
from models.model import GeminiModel
from models.model import KimiModel
from models.model import ZhipuModel
from models.model import DoubaoModel
from models.model import XunfeiModel
from models.model import HunyuanModel
from models.model import WenxinModel
from games.guess import Guess70PercentGame
from games.guess_known import AdaptiveGuessGame
import config
def guess():
    # 初始化模型
    models = [
        DeepSeekModel(config.DEEPSEEK_KEY, "DeepSeek-R1"),
        GeminiModel(config.GEMINI_KEY, "gemini-1.5-pro"),
        KimiModel(config.KIMI_KEY, "moonshot-v1-32k"),
        ZhipuModel(config.ZHIPU_KEY, "glm-4-flash"),
        DoubaoModel(config.DOUBAO_KEY, "doubao-1-5-pro-32k-250115"),
        XunfeiModel(config.XUNFEI_KEY, "4.0Ultra"),
        HunyuanModel(config.HUNYUAN_KEY, "hunyuan-turbo"),
        WenxinModel(config.BAIDU_KEY, "WGWaPA7PS0OqY3WvawLTGgN1PjfTNfDw","ernie-4.0-turbo-8k-latest")
    ]
    players = [
        DeepSeekModel(config.DEEPSEEK_KEY, "DeepSeek-R1"),
        GeminiModel(config.GEMINI_KEY, "gemini-1.5-pro"),
        KimiModel(config.KIMI_KEY, "moonshot-v1-32k"),
        ZhipuModel(config.ZHIPU_KEY, "glm-4-flash"),
        DoubaoModel(config.DOUBAO_KEY, "doubao-1-5-pro-32k-250115"),
        XunfeiModel(config.XUNFEI_KEY, "4.0Ultra"),
        HunyuanModel(config.HUNYUAN_KEY, "hunyuan-turbo"),
        WenxinModel(config.BAIDU_KEY, "WGWaPA7PS0OqY3WvawLTGgN1PjfTNfDw", "ernie-4.0-turbo-8k-latest")
    ]

    # 初始化游戏
    game = AdaptiveGuessGame(models=players, max_rounds=5)
    # game = Guess70PercentGame(models=players, max_rounds=5)
    # 运行博弈实验
    results = game.run_game()


    # 展示结果
    print("\n最终结果：")
    for round_result in results:
        print(f"\n第 {round_result['round']} 轮：")
        print(f"目标值：{round_result['target']:.2f}")
        for model, guess in round_result['guesses'].items():
            status = "√" if model == round_result['winner'] else ""
            print(f"{model}: {guess} {status}")