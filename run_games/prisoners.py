from models.model import OpenAIModel
from models.model import DeepSeekModel
from models.model import GeminiModel
from models.model import KimiModel
from models.model import ZhipuModel
from models.model import DoubaoModel
from models.model import XunfeiModel
from models.model import HunyuanModel
from models.model import WenxinModel
from games.prisoners_dilemma import PrisonersDilemmaGame
from games.prisoners_dilemma import HumanPlayer
from games.parallel_prisoners_dilemma import ParallelPrisonersDilemma
import config

def prisoners():
    # 初始化模型
    DeepSeek = DeepSeekModel(config.DEEPSEEK_KEY, "DeepSeek-R1")
    Gemini = GeminiModel(config.GEMINI_KEY, "gemini-1.5-pro")
    Kimi = KimiModel(config.KIMI_KEY, "moonshot-v1-32k")
    Zhipu = ZhipuModel(config.ZHIPU_KEY, "glm-4-flash")
    Doubao = DoubaoModel(config.DOUBAO_KEY, "doubao-1-5-pro-32k-250115")
    Xunfei = XunfeiModel(config.XUNFEI_KEY, "4.0Ultra")
    Hunyuan = HunyuanModel(config.HUNYUAN_KEY, "hunyuan-turbo")
    Wenxin = WenxinModel(config.BAIDU_KEY, "WGWaPA7PS0OqY3WvawLTGgN1PjfTNfDw", "ernie-4.0-turbo-8k-latest")

    players = [Kimi, Hunyuan]

    # player_pairs = [
    #     [DeepSeek, Xunfei],
    #     [Kimi, Hunyuan],
    #     [Doubao, Wenxin],
    #     [Gemini, Zhipu]
    # ]

    player_pairs = [
        [DeepSeek, Doubao],
        [Gemini, Hunyuan],
        [Kimi, Wenxin],
        [Xunfei, Zhipu]
    ]

    # 初始化游戏
    # game = PrisonersDilemmaGame(players=players, rounds=10, communication_start_round=False, prompt_type='abstract')
    parallel_game = ParallelPrisonersDilemma(
        player_pairs=player_pairs,
        rounds=10,
        communication_start_round=6,
        prompt_type="abstract",
        save_results=True,
        results_dir="game_results"
    )
    # 运行博弈实验
    # history = game.run_game()
    results = parallel_game.run_all_games(max_parallel=4)
