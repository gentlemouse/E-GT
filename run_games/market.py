from models.model import OpenAIModel
from models.model import DeepSeekModel
from models.model import GeminiModel
from models.model import KimiModel
from models.model import ZhipuModel
from models.model import DoubaoModel
from models.model import XunfeiModel
from models.model import HunyuanModel
from models.model import WenxinModel
from games.market_entry_game import print_game_history
from games.market_entry_game import MarketEntryGame
from games.market_entry_game import EnhancedMarketEntryGame
from games.market_extensions import AsymmetricProfitCalculator, CoalitionManager, PunishmentMechanism, ImperfectInformationProvider
import config

def market():
    models = [
        DeepSeekModel(config.DEEPSEEK_KEY, "DeepSeek-R1"),
        GeminiModel(config.GEMINI_KEY, "gemini-1.5-pro"),
        KimiModel(config.KIMI_KEY, "moonshot-v1-8k"),
        ZhipuModel(config.ZHIPU_KEY, "glm-4v-flash"),
        DoubaoModel(config.DOUBAO_KEY, "doubao-1-5-lite-32k-250115"),
        XunfeiModel(config.XUNFEI_KEY, "lite"),
        HunyuanModel(config.HUNYUAN_KEY, "hunyuan-lite"),
        WenxinModel(config.BAIDU_KEY, "WGWaPA7PS0OqY3WvawLTGgN1PjfTNfDw", "ernie-4.0-turbo-8k")
    ]
    participants = [
        XunfeiModel(config.XUNFEI_KEY, "lite"),
        HunyuanModel(config.HUNYUAN_KEY, "hunyuan-lite"),
        WenxinModel(config.BAIDU_KEY, "WGWaPA7PS0OqY3WvawLTGgN1PjfTNfDw", "ernie-4.0-turbo-8k")
    ]

    game = MarketEntryGame(
        models=participants,
        max_rounds=2,
        market_capacity=2,
        base_profit=150,
        decline_rate=0.25
    )
    game.run_game()
    print_game_history(game)

    # player_factors = {
    #     "GPT-4": {"cost_factor": 0.7, "brand_power": 1.5},
    #     "Gemini": {"cost_factor": 0.9, "brand_power": 0.8},
    #     "Claude": {"cost_factor": 1.0, "brand_power": 1.0}
    # }
    # coalition_config = {
    #     "AI联盟": ["GPT-4", "Gemini"]
    # }
    # game = EnhancedMarketEntryGame(
    #     models=participants,
    #     max_rounds=2,
    #     market_capacity=2,
    #     base_profit=150,
    #     decline_rate=0.25,
    #     # profit_calculator=AsymmetricProfitCalculator(player_factors),
    #     # coalition_manager=CoalitionManager(coalition_config),
    #     punishment=PunishmentMechanism(0.3),
    #     info_provider=ImperfectInformationProvider
    # )
    #
    # # 运行游戏
    # history = game.run_game()
