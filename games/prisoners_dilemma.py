import re
import random
import textwrap
from typing import List, Dict, Tuple, Union
from datetime import datetime
from models.base_model import BaseModel
from games.prompts.prisoners_dilemma_prompts import (
    generate_prompt, 
    generate_human_prompt
)


class HumanPlayer(BaseModel):
    """人类玩家交互处理类"""
    def __init__(self, name="人类玩家"):
        # 添加api_key参数（虽然人类玩家不需要）
        super().__init__(api_key="human_player", model_name=name)
        self.require_input = True

    def generate_response(self, prompt: str) -> str:
        """处理人类玩家的输入"""
        print("\n" + "=" * 40)
        print(prompt)
        return input("请输入你的决策（直接输入合作/背叛）：").strip()


class PrisonersDilemmaGame:
    def __init__(
            self,
            players: List[Union[BaseModel, HumanPlayer]],  # 修改支持人类玩家
            rounds: int = 5,
            payoff_matrix: Dict[str, Tuple[int, int]] = None,
            communication_start_round: Union[bool, int] = False,
            prompt_type: str = "normal",
            wait_for_input: bool = True  # 添加参数控制是否等待用户输入
    ):
        assert len(players) == 2, "必须且只能有两个参与者"
        assert sum(isinstance(p, HumanPlayer) for p in players) <= 1, "最多一个人类玩家"
        assert prompt_type in ["normal", "single", "opposite", "abstract"], "提示词类型必须是'normal'、'single'、'opposite'或'abstract'"
        self.players = {p.model_name: p for p in players}
        self.rounds = rounds
        self.history = []
        self.human_player = next((p for p in players if isinstance(p, HumanPlayer)), None)
        self.prompt_type = prompt_type
        self.wait_for_input = wait_for_input  # 是否等待用户输入

        # 处理沟通参数
        if isinstance(communication_start_round, bool):
            self.communication_start = 0 if communication_start_round else None
        else:
            self.communication_start = max(0, int(communication_start_round))

        self.payoff = payoff_matrix or {
            ('合作', '合作'): (8, 8),
            ('合作', '背叛'): (-10, 20),
            ('背叛', '合作'): (20, -10),
            ('背叛', '背叛'): (-5, -5)
        }

    def _generate_prompt(self, player: str, opponent: str, history: List, message: str = None) -> str:
        if self.players[player] == self.human_player:
            return generate_human_prompt(opponent, history, self.rounds, self.payoff, message, self.prompt_type)
        
        # 使用导入的提示词生成函数
        return generate_prompt(
            self.prompt_type, 
            player, 
            opponent, 
            history, 
            self.rounds, 
            self.payoff, 
            message
        )

    def _parse_response(self, text: str, is_human: bool = False) -> Tuple[str, str]:
        """解析响应，区分人类和AI"""
        if is_human:
            decision = self._validate_human_decision(text)
            return "人类玩家直接决策", decision

        try:
            thought = re.search(r"<思考>(.*?)</思考>", text, re.DOTALL)
            
            # 根据prompt_type选择不同的正则表达式模式
            if self.prompt_type == "abstract":
                decision_match = re.search(
                    r"<决策>\s*\[?\s*(动作1|动作2)\s*\]?\s*</决策>",
                    text,
                    re.IGNORECASE
                )
                # 将"动作1"、"动作2"转换回"合作"、"背叛"
                action_map = {
                    "动作1": "合作",
                    "动作2": "背叛"
                }
                clean_thought = thought.group(1).strip() if thought else "未提供策略分析"
                if decision_match:
                    abstract_decision = decision_match.group(1).strip()
                    clean_decision = action_map.get(abstract_decision, self._fallback_decision(text))
                else:
                    clean_decision = self._fallback_decision(text)
            else:
                decision_match = re.search(
                    r"<决策>\s*\[?\s*(合作|背叛)\s*\]?\s*</决策>",
                    text,
                    re.IGNORECASE
                )
                clean_thought = thought.group(1).strip() if thought else "未提供策略分析"
                clean_decision = decision_match.group(1).strip() if decision_match else self._fallback_decision(text)
                clean_decision = "合作" if clean_decision.lower() in ["合作", "coop"] else "背叛"

            return clean_thought, clean_decision
        except Exception as e:
            print(f"解析错误: {str(e)}")
            return "响应格式异常", random.choice(["合作", "背叛"])

    def _validate_human_decision(self, input_text: str) -> str:
        """验证人类玩家输入"""
        clean_text = input_text.strip().lower()
        
        # 处理abstract模式下的输入
        if self.prompt_type == "abstract":
            if clean_text in ["动作1", "action1", "a1"]:
                return "合作"
            if clean_text in ["动作2", "action2", "a2"]:
                return "背叛"
        # 处理普通模式下的输入
        else:
            if clean_text in ["合作", "coop", "c"]:
                return "合作"
            if clean_text in ["背叛", "defect", "d"]:
                return "背叛"
                
        print("无效输入，请重新选择")
        return self._validate_human_decision(input("\n请选择【合作/背叛】或【动作1/动作2】："))

    def _fallback_decision(self, text: str) -> str:
        """备选决策解析"""
        text = text.lower()
        
        if self.prompt_type == "abstract":
            coop_keywords = ["动作1", "action1", "a1"]
            defect_keywords = ["动作2", "action2", "a2"]
        else:
            coop_keywords = ["合作", "coop", "yes", "agree", "accept"]
            defect_keywords = ["背叛", "defect", "no", "deny", "refuse"]

        if any(kw in text for kw in coop_keywords):
            return "合作"
        if any(kw in text for kw in defect_keywords):
            return "背叛"
        return random.choice(["合作", "背叛"])

    def _calculate_payoff(self, decision_a: str, decision_b: str) -> Tuple[int, int]:
        if self.prompt_type == "opposite":
            # 反转得分规则
            opposite_payoff = {
                ('合作', '合作'): (-5, -5),    # 双方合作 → 各得-5分
                ('合作', '背叛'): (20, -10),   # 你合作对方背叛 → 你得20分，对方得-10分
                ('背叛', '合作'): (-10, 20),   # 你背叛对方合作 → 你得-10分，对方得20分
                ('背叛', '背叛'): (8, 8)       # 双方背叛 → 各得8分
            }
            return opposite_payoff.get((decision_a, decision_b), (0, 0))
        else:
            # 使用正常得分规则（normal、single、abstract等类型）
            return self.payoff.get((decision_a, decision_b), (0, 0))

    def play_round(self, round_num: int) -> Dict:
        p1, p2 = list(self.players.keys())
        messages = {}

        # 处理沟通阶段
        if self.communication_start is not None and round_num > self.communication_start:
            for player in [p1, p2]:
                if self.players[player] == self.human_player:
                    msg = input("\n发送给对手的消息（30字内，回车跳过）：")[:30]
                    messages[player] = msg.strip() or "无消息"
                else:
                    prompt = f"请发送给对手的协商消息（30字以内）："
                    response = self.players[player].generate_response(prompt)
                    messages[player] = response[:30].strip()

        # 收集决策
        decisions = {}
        for player, opponent in [(p1, p2), (p2, p1)]:
            is_human = self.players[player] == self.human_player
            opponent_message = messages.get(opponent, None)

            if is_human:
                prompt = self._generate_prompt(player, opponent, self.history, opponent_message)
                print("\n" + "=" * 40)
                print(prompt)
                if self.prompt_type == "abstract":
                    raw_input = input("请输入你的决策（直接输入动作1/动作2）：").strip()
                else:
                    raw_input = input("请输入你的决策（直接输入合作/背叛）：").strip()
            else:
                prompt = self._generate_prompt(player, opponent, self.history, opponent_message)
                raw_input = self.players[player].generate_response(prompt)

            thought, decision = self._parse_response(raw_input, is_human=is_human)

            decisions[player] = {
                "decision": decision,
                "thought": thought,
                "raw_response": raw_input
            }

        # 计算得分
        scores = self._calculate_payoff(decisions[p1]["decision"], decisions[p2]["decision"])

        round_data = {
            "round": round_num,
            "decisions": decisions,
            "scores": {p1: scores[0], p2: scores[1]},
            "messages": messages if messages else None,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(round_data)
        return round_data

    def run_game(self):
        communication_status = "关闭" if self.communication_start is None else \
            f"从第{self.communication_start + 1}轮开启" if self.communication_start > 0 else "全程开启"

        print(f"\n{'=' * 30}")
        print(f" 合作博弈实验启动 ")
        print(f" 参赛模型：{' vs '.join(self.players.keys())}")
        print(f" 总轮次：{self.rounds} | 沟通机制：{communication_status}")
        print(f"{'=' * 30}\n")

        try:
            for current_round in range(1, self.rounds + 1):
                print(f"\n{'=' * 20} 第 {current_round} 轮 {'=' * 20}")
                round_data = self.play_round(current_round)
                self._print_round_result(round_data)

                # 只有在wait_for_input为True且不是最后一轮时才等待用户输入
                if self.wait_for_input and current_round < self.rounds:
                    user_input = input("\n按回车继续，或输入q退出: ")
                    if user_input.lower() == 'q':
                        print("实验提前终止")
                        break
        finally:
            self._analyze_outcome()

        return self.history

    def _print_round_result(self, round_data: Dict):
        p1, p2 = list(self.players.keys())

        print(f"\n[回合 {round_data['round']} 结果]")
        print(f"{'-' * 40}")

        self._print_player_decision(p1, round_data)
        print(f"{'-' * 40}")
        self._print_player_decision(p2, round_data)
        print(f"{'-' * 40}")

        print(
            f"累计得分：{p1}={sum(r['scores'][p1] for r in self.history)} | {p2}={sum(r['scores'][p2] for r in self.history)}")

        if round_data['messages']:
            print(f"\n协商消息：")
            print(f"  {p1}: {round_data['messages'].get(p1, '无')}")
            print(f"  {p2}: {round_data['messages'].get(p2, '无')}")

    def _print_player_decision(self, player: str, round_data: Dict):
        data = round_data['decisions'][player]
        print(f"{player} 决策分析：")
        print(textwrap.fill(f"思考过程：{data['thought']}", width=80, initial_indent='  ', subsequent_indent='  '))
        print(f"最终选择：{data['decision']}")
        print(f"本轮得分：+{round_data['scores'][player]}")

    def _analyze_outcome(self):
        p1, p2 = list(self.players.keys())

        print(f"\n{'=' * 30}")
        print(f" 最终博弈分析报告 ")
        print(f"{'=' * 30}")

        total_scores = {
            p1: sum(r["scores"][p1] for r in self.history),
            p2: sum(r["scores"][p2] for r in self.history)
        }
        # cooperation_rate = {
        #     p1: sum(1 for r in self.history if r["decisions"][p1]["decision"] == "合作") / len(self.history),
        #     p2: sum(1 for r in self.history if r["decisions"][p2]["decision"] == "合作") / len(self.history)
        # }

        print(f"\n总得分：")
        print(f"  {p1}: {total_scores[p1]} 分")
        print(f"  {p2}: {total_scores[p2]} 分")

        # print(f"\n合作率：")
        # print(f"  {p1}: {cooperation_rate[p1]:.1%}")
        # print(f"  {p2}: {cooperation_rate[p2]:.1%}")

        # print(f"\n策略模式识别：")
        # self._analyze_strategy_pattern(p1)
        # self._analyze_strategy_pattern(p2)

        # print(f"\n原始数据已存档：")
        # print(f"  - 总轮次：{len(self.history)}")
        # print(f"  - 时间范围：{self.history[0]['timestamp']} 至 {self.history[-1]['timestamp']}")

    def _analyze_strategy_pattern(self, player: str):
        pattern = ''.join(['C' if r["decisions"][player]["decision"] == "合作" else 'D' for r in self.history])
        strategy_type = "未知"
        if pattern.count('C') == len(pattern):
            strategy_type = "绝对合作型"
        elif pattern.count('D') == len(pattern):
            strategy_type = "绝对背叛型"
        elif 'DDC' in pattern:
            strategy_type = "报复型（一报还一报）"
        elif pattern.endswith('CC'):
            strategy_type = "宽容型"
        elif 'CD' in pattern and 'DC' in pattern:
            strategy_type = "随机试探型"
        print(f"  {player}：{strategy_type}")
        print(f"    决策序列：{pattern}")


# 使用示例
'''
if __name__ == "__main__":
    from models.openai_model import OpenAIModel
    from models.gemini_model import GeminiModel
    import config

    # 初始化参与者
    human = HumanPlayer("玩家小明")
    ai = OpenAIModel(config.OPENAI_KEY, "GPT-4")

    # 运行博弈实验（前2轮不沟通，第3轮开始沟通）
    game = PrisonersDilemmaGame(
        players=[human, ai],
        rounds=5,
        communication_start_round=2
    )
    history = game.run_game()
'''