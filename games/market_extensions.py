import random
from typing import Dict, List, Tuple
import numpy as np

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