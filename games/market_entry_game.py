import numpy as np
import random
import re
from typing import List, Dict, Tuple
from datetime import datetime
from models.base_model import BaseModel
from utils.response_parser import parse_decision


class MarketEntryGame:
    def __init__(
            self,
            models: List[BaseModel],
            max_rounds: int = 5,
            market_capacity: int = 5,  # 市场最大盈利进入者数量
            base_profit: float = 100.0,  # 基准利润
            decline_rate: float = 0.2,  # 利润递减率
    ):
        """
        市场进入博弈模拟器

        参数说明：
            market_capacity: 市场饱和临界点（N）
            base_profit: 市场总基准利润（当刚好N个进入者时总利润）
            decline_rate: 边际利润递减率（每超出一个进入者的利润衰减比例）
        """
        self.models = models
        self.max_rounds = max_rounds
        self.params = {
            'N': market_capacity,
            'P0': base_profit,
            'α': decline_rate
        }
        self.history = []
        self.current_round = 1
        self._init_profit_matrix()

    def _init_profit_matrix(self):
        """预计算所有可能进入者数量的利润情况"""
        self.profit_cache = {}
        max_possible_entrants = len(self.models)

        for k in range(max_possible_entrants + 1):
            if k <= self.params['N']:
                profit = self.params['P0'] * (1 - self.params['α']) ** max(k - self.params['N'], 0)
            else:
                profit = 0
            self.profit_cache[k] = profit

    def _calculate_profit(self, entrants_count: int) -> float:
        if entrants_count > self.params['N']:
            return 0
        else:
            return self.params['P0'] * (1 - self.params['α']) ** (entrants_count-1)

    def _build_market_report(self) -> str:
        if not self.history:
            return "全新市场，暂无历史数据"

        report = ["历史市场状况："]
        for rd in self.history:
            report.append(f"第 {rd['round']} 轮：")
            report.append(f"  - 进入者数量：{rd['entrants_count']}家")
            report.append(f"  - 单个企业利润：${rd['profit_per_entrant']:.2f}")
            report.append(f"  - 总市场利润：${rd['total_profit']:.2f}")
        return "\n".join(report)

    def _generate_prompt(self, model: BaseModel) -> str:
        market_status = self._build_market_report()
        current_round = self.current_round

        return f"""
        您是一家企业的战略决策AI，正在为第 {current_round} 轮做出决策。
        请根据以下信息做出决策：

        {market_status}

        市场规则：
        1. 当进入企业 ≤ {self.params['N']} 家时，每家利润为 ${self.params['P0']} × (1 - {self.params['α']})^(超额数量)
        2. 当进入企业 > {self.params['N']} 家时，所有企业将亏损
        3. 您需要与其他{len(self.models) - 1}个竞争者同时做出决策

        请严格按照以下格式回应：
        <决策>
        [enter/exit]
        </决策>

        <理由>
        （简要分析市场状况和预期收益）
        </理由>
        """

    def _parse_decisions(self, responses: Dict[str, str]) -> Tuple[List[str], List[str]]:
        entrants = []
        non_entrants = []

        for model_name, response in responses.items():
            decision = parse_decision(response)
            if decision == "进入":
                entrants.append(model_name)
            else:
                non_entrants.append(model_name)

        return entrants, non_entrants

    def play_round(self) -> Dict:
        """执行单轮博弈，并存储提示词"""
        round_data = {
            "round": self.current_round,
            "decisions": {},
            "entrants_count": 0,
            "profit_per_entrant": 0.0,
            "total_profit": 0.0,
            "timestamp": datetime.now().isoformat(),
            "prompt": None  # 新增字段用于存储提示词
        }

        # 生成并存储提示词（假设所有模型使用相同的提示词）
        prompt = self._generate_prompt(self.models[0])  # 用第一个模型生成提示词
        round_data["prompt"] = prompt

        # 收集决策
        responses = {}
        for model in self.models:
            try:
                response = model.generate_response(prompt)
                responses[model.model_name] = response
            except Exception as e:
                print(f"{model.model_name} 决策失败: {str(e)}")
                responses[model.model_name] = ""

        # 解析决策
        entrants, non_entrants = self._parse_decisions(responses)
        entrants_count = len(entrants)
        profit = self._calculate_profit(entrants_count)

        # 记录结果
        round_data.update({
            "decisions": {
                name: {
                    "raw_response": responses[name],
                    "decision": "进入" if name in entrants else "不进入",
                    "profit": profit if name in entrants else 0
                } for name in responses
            },
            "entrants_count": entrants_count,
            "profit_per_entrant": profit,
            "total_profit": profit * entrants_count
        })

        self.history.append(round_data)
        self.current_round += 1
        return round_data

    def run_game(self):
        """运行完整博弈"""
        print(f"市场进入博弈开始｜最大轮次：{self.max_rounds}｜参赛企业：{len(self.models)}")

        while self.current_round <= self.max_rounds:
            print(f"\n=== 第 {self.current_round} 轮 ===")
            round_data = self.play_round()
            self._print_round_summary(round_data)

        self._analyze_strategies()
        return self.history

    def _print_round_summary(self, round_data: Dict):
        """打印轮次摘要"""
        print(f"进入企业：{round_data['entrants_count']}家")
        print(f"单企利润：${round_data['profit_per_entrant']:.2f}")
        print("企业决策详情：")
        for name, data in round_data["decisions"].items():
            symbol = "🚪" if data["decision"] == "进入" else "🚫"
            print(f"{symbol} {name}: {data['decision']} (利润: ${data['profit']:.2f})")

    def _analyze_strategies(self):
        """生成策略分析报告"""
        print("\n=== 博弈策略分析 ===")

        # 计算纳什均衡偏离度
        theoretical_equilibrium = min(self.params['N'], len(self.models))
        actual_entrants = [rd["entrants_count"] for rd in self.history]
        avg_deviation = np.mean(np.abs(np.array(actual_entrants) - theoretical_equilibrium))

        print(f"理论纳什均衡：{theoretical_equilibrium}家进入")
        print(f"实际平均偏离：{avg_deviation:.2f}家")

        # 识别最优策略者
        profitability = {}
        for model in self.models:
            total_profit = sum(
                rd["decisions"][model.model_name]["profit"]
                for rd in self.history
            )
            profitability[model.model_name] = total_profit

        best_performer = max(profitability, key=profitability.get)
        print(f"\n最佳表现者：{best_performer}（总利润：${profitability[best_performer]:.2f}）")

        # 绘制趋势图
        self._plot_entrant_trend(actual_entrants)

    def _plot_entrant_trend(self, actual_entrants: List[int]):
        """使用ASCII绘制进入者趋势图"""
        print("\n进入者数量趋势：")
        max_count = max(actual_entrants + [self.params['N']])

        for i, count in enumerate(actual_entrants, 1):
            bar = "▇" * count + " " * (max_count - count)
            equilibrium_mark = "★" if count == self.params['N'] else ""
            print(f"第{i}轮: [{bar}] {count}家{equilibrium_mark}")


def extract_reasoning(response: str) -> str:
    """从模型响应中提取理由部分"""
    pattern = r"<理由>(.*?)</理由>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "未提供理由"


def print_game_history(game: MarketEntryGame):
    """打印每轮的提示词和每个模型的理由"""
    print("\n=== 游戏历史记录 ===")
    for round_data in game.history:
        print(f"\n第 {round_data['round']} 轮提示词：")
        print(f"{round_data['prompt']}")
        print(f"\n各模型理由：")
        for model_name, data in round_data["decisions"].items():
            reasoning = extract_reasoning(data["raw_response"])
            print(f"{model_name}: {reasoning}")
        print("-" * 50)


'''
if __name__ == "__main__":
    from models.openai_model import OpenAIModel
    from models.gemini_model import GeminiModel
    import config

    participants = [
        OpenAIModel(config.OPENAI_KEY, "gpt-4-turbo"),
        GeminiModel(config.GEMINI_KEY),
        # 添加其他模型...
    ]

    game = MarketEntryGame(
        models=participants,
        market_capacity=3,
        base_profit=150,
        decline_rate=0.25
    )
    game.run_game()
'''

class EnhancedMarketEntryGame(MarketEntryGame):
    def __init__(self, *args, **kwargs):
        # 非对称利润配置
        self.profit_calculator = kwargs.pop('profit_calculator', None)
        # 不完全信息配置
        self.info_provider = kwargs.pop('info_provider', ImperfectInformationProvider())
        # 惩罚机制配置
        self.punishment_system = kwargs.pop('punishment', PunishmentMechanism())
        # 合作联盟配置
        self.coalition_manager = kwargs.pop('coalition_manager', None)

        super().__init__(*args, **kwargs)
        self._init_profit_matrix()

    def _build_market_report(self) -> str:
        """重写市场报告生成"""
        return self.info_provider.obscure_information(self.history, self.current_round)

    def _calculate_profit(self, entrant: str, entrants_count: int) -> float:
        """重写利润计算"""
        base_profit = super()._calculate_profit(entrants_count)
        if self.profit_calculator:
            return self.profit_calculator.calculate_profit(entrant, base_profit, entrants_count)
        return base_profit

    def play_round(self) -> Dict:
        """扩展决策流程"""
        # 合作联盟决策
        if self.coalition_manager:
            coalitions = self.coalition_manager.form_coalitions([m.model_name for m in self.models])
            for coalition in coalitions.values():
                joint_decision = self.coalition_manager.make_joint_decision(coalition, self.history)
                # 覆盖联盟成员的决策
                for member in coalition:
                    # 这里简化处理，实际需要修改模型决策逻辑
                    print(f"联盟{coalition}决定{joint_decision}")

        round_data = super().play_round()

        # 应用惩罚机制
        self.punishment_system.apply_punishment(self.history)
        for name in round_data['decisions']:
            punishment = self.punishment_system.get_punishment(name, self.current_round)
            round_data['decisions'][name]['profit'] *= (1 - punishment)

        return round_data


# 使用示例
'''if __name__ == "__main__":
    from models.openai_model import OpenAIModel
    from models.gemini_model import GeminiModel
    import config

    participants = [
        OpenAIModel(config.OPENAI_KEY, "gpt-4-turbo"),
        GeminiModel(config.GEMINI_KEY),
        # 添加其他模型...
    ]

    game = MarketEntryGame(
        models=participants,
        market_capacity=3,
        base_profit=150,
        decline_rate=0.25
    )
    game.run_game()
    
# 初始化复杂场景
模块	        功能特点	            实现机制
非对称利润	企业差异化成本/品牌溢价	基于企业属性动态调整利润公式
不完全信息	模糊历史数据/部分可见	概率性隐藏信息+数据扰动
惩罚机制	    超额进入惩罚/跨期惩罚	追踪历史行为+利润扣减
合作博弈	    联盟决策/协同策略	    预定义联盟+集体决策覆盖

非对称利润
player_factors = {
    "GPT-4": {"cost_factor": 0.7, "brand_power": 1.5},
    "Gemini": {"cost_factor": 0.9, "brand_power": 0.8},
    "Claude": {"cost_factor": 1.0, "brand_power": 1.0}
}
合作博弈
coalition_config = {
    "AI联盟": ["GPT-4", "Gemini"]
}

game = EnhancedMarketEntryGame(
    models=models,
    market_capacity=3,
    profit_calculator=AsymmetricProfitCalculator(player_factors),
    coalition_manager=CoalitionManager(coalition_config),
    punishment=PunishmentMechanism(0.3)
    info_provider=ImperfectInformationProvider
)

# 运行游戏
history = game.run_game()'''

class AsymmetricProfitCalculator:
    """
    非对称利润计算模块（企业差异化）
    使用方法：在MarketEntryGame初始化时传入
    """

    def __init__(self, player_factors: Dict[str, Dict]):
        """
        :param player_factors: 各企业差异参数
        示例：{
            "GPT-4": {"cost_factor": 0.8, "brand_power": 1.2},
            "Gemini": {"cost_factor": 1.0, "brand_power": 1.0}
        }
        """
        self.player_factors = player_factors

    def calculate_profit(self, entrant: str, base_profit: float, entrants_count: int) -> float:
        """计算差异化利润"""
        factors = self.player_factors.get(entrant, {})
        adjusted_profit = base_profit * factors.get("brand_power", 1.0)
        cost = base_profit * (1 - factors.get("cost_factor", 1.0))
        return max(0, adjusted_profit - cost)


class ImperfectInformationProvider:
    """
    不完全信息博弈模块
    使用方法：继承并重写_build_market_report方法
    """

    def obscure_information(self, history: List[Dict], current_round: int) -> str:
        """生成模糊的市场报告"""
        if not history:
            return "市场信息不透明，无可靠历史数据"

        # 随机隐藏部分历史
        visible_rounds = [rd for rd in history if random.random() < 0.7]
        if not visible_rounds:
            return "未能获取有效市场情报"

        report = []
        for rd in visible_rounds[-2:]:  # 最多显示最近2轮
            profit_desc = "盈利" if rd['profit_per_entrant'] > 50 else "竞争激烈"
            report.append(
                f"第{rd['round']}轮：进入者约{rd['entrants_count']}±{random.randint(1, 2)}家，{profit_desc}"
            )
        return "模糊市场情报：\n" + "\n".join(report)


class PunishmentMechanism:
    """
    重复博弈惩罚机制模块
    使用方法：在每轮结束后调用apply_punishment
    """

    def __init__(self, punishment_factor=0.2):
        self.punishment_records = {}  # {offender: {round: punishment}}
        self.punishment_factor = punishment_factor

    def apply_punishment(self, history: List[Dict]):
        """应用惩罚逻辑"""
        if len(history) < 2:
            return

        last_round = history[-2]
        entrants = [name for name, data in last_round['decisions'].items() if data['decision'] == '进入']

        # 如果上轮超额进入，惩罚所有进入者
        if last_round['entrants_count'] > last_round['params']['N']:
            for entrant in entrants:
                if entrant not in self.punishment_records:
                    self.punishment_records[entrant] = {}
                self.punishment_records[entrant][len(history)] = self.punishment_factor

    def get_punishment(self, player: str, current_round: int) -> float:
        """获取当前惩罚系数"""
        return sum(
            factor
            for rnd, factor in self.punishment_records.get(player, {}).items()
            if rnd >= current_round - 2  # 惩罚持续2轮
        )


class CoalitionManager:
    """
    合作博弈场景模块
    使用方法：在决策前调用form_coalitions
    """

    def __init__(self, coalition_config: Dict[str, List[str]]):
        """
        :param coalition_config: 联盟配置{"coalition1": ["GPT-4", "Gemini"]}
        """
        self.coalitions = coalition_config

    def form_coalitions(self, players: List[str]) -> Dict[str, List[str]]:
        """形成合作联盟（简化版）"""
        active_coalitions = {}
        for name, members in self.coalitions.items():
            # 检查是否所有成员都参与本轮游戏
            if all(m in players for m in members):
                active_coalitions[name] = members
        return active_coalitions

    def make_joint_decision(self, coalition: List[str], history: List[Dict]) -> str:
        """联盟集体决策"""
        # 简单策略：当平均历史利润超过阈值时集体进入
        avg_profit = np.mean([rd['profit_per_entrant'] for rd in history]) if history else 0
        return "进入" if avg_profit > 50 else "不进入"