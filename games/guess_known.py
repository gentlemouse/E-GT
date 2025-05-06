import numpy as np
import re
from typing import List, Dict, Optional
from datetime import datetime
from models.base_model import BaseModel
from utils.response_parser import extract_number


class AdaptiveGuessGame:
    def __init__(
            self,
            models: List[BaseModel],
            max_rounds: int = 3,
            min_value: float = 0,
            max_value: float = 100
    ):
        self.models = models
        self.max_rounds = max_rounds
        self.min_value = min_value
        self.max_value = max_value
        self.history = []
        self.current_round = 1

    def _calculate_target(self, guesses: List[float]) -> float:
        valid_guesses = [g for g in guesses if g is not None]
        return np.mean(valid_guesses) * 0.7 if valid_guesses else 0

    def _build_history_context(self, model_name: str) -> str:
        """构建历史信息上下文"""
        context = []
        for round_data in self.history:
            user_guess = round_data["guesses"][model_name]["parsed_value"]
            context.append(
                f"第{round_data['round']}轮结果：\n"
                f"- 平均值：{round_data['average']:.2f}\n"
                f"- 目标值：{round_data['target']:.2f}\n"
                f"- 你的猜测：{user_guess if user_guess else '无效'}"
            )
        return "\n\n".join(context) if context else "无历史数据"

    def _generate_prompt(self, model_name: str) -> str:
        """生成包含历史信息的提示语"""
        # base_prompt = (
        #     f"请猜测当前轮所有参与者猜测数字平均值的70%\n"
        #     f"数值范围：{self.min_value}-{self.max_value}\n"
        #     "格式要求：\n"
        #     "1. 只返回最终数值\n"
        #     "2. 使用阿拉伯数字格式（如：42 或 56.7）\n"
        #     "3. 不要包含任何解释\n"
        #     f"当前是第{self.current_round}轮，请给出数字："
        # )
        base_prompt = (
            f"请猜测当前轮所有参与者猜测数字平均值的70%，数值范围：{self.min_value}-{self.max_value}，当前是第{self.current_round}轮"
            "格式要求：请严格遵守回答格式：先说明思路，无论思路如何必须在最后给出一个确切的数字答案，例如：我认为...所以我给出的最终答案是...（一定保证你回答的最后字符是数字答案）"
        )
        history_section = self._build_history_context(model_name)

        return (
            f"{base_prompt}\n\n"
            "历史信息：\n"
            f"{history_section}\n\n"
            "你的最终猜测数值："
        )

    def play_round(self) -> Dict:
        round_result = {
            "round": self.current_round,
            "guesses": {},
            "average": None,
            "target": None,
            "winner": None,
            "timestamp": datetime.now().isoformat()
        }

        # 收集所有模型的猜测
        guesses = {}
        for model in self.models:
            try:
                prompt = self._generate_prompt(model.model_name)
                response = model.generate_response(prompt)
                raw_guess = extract_number(response)

                round_result["guesses"][model.model_name] = {
                    "raw_response": response,
                    "parsed_value": raw_guess
                }
                guesses[model.model_name] = raw_guess

                # 数值处理
                if raw_guess is not None:
                    processed_guess = max(self.min_value, min(self.max_value, raw_guess))
                    processed_guess = round(processed_guess, 2)
                else:
                    processed_guess = None
                    print(f"[WARNING] {model.model_name} 无效响应: {response}")

                round_result["guesses"][model.model_name] = {
                    "raw_prompt": prompt,
                    "raw_response": response,
                    "parsed_value": processed_guess
                }
                guesses[model.model_name] = processed_guess
            except Exception as e:
                print(f"[ERROR] {model.model_name} 猜测失败: {str(e)}")
                guesses[model.model_name] = None

        # 计算统计值
        valid_guesses = [g for g in guesses.values() if g is not None]
        current_average = np.mean(valid_guesses) if valid_guesses else 0
        target = self._calculate_target(valid_guesses)

        # 确定胜者
        closest = None
        min_diff = float('inf')
        for name, guess in guesses.items():
            if guess is None:
                continue
            diff = abs(guess - target)
            if diff < min_diff:
                min_diff = diff
                closest = name

        round_result.update({
            "average": round(current_average, 2),
            "target": round(target, 2),
            "winner": closest
        })

        self.history.append(round_result)
        self.current_round += 1
        return round_result

    def run_game(self):
        print(f"自适应游戏开始（{self.max_rounds}轮）")
        while self.current_round <= self.max_rounds:
            print(f"\n=== 第 {self.current_round} 轮 ===")
            round_data = self.play_round()
            self._print_round_summary(round_data)

        self._print_final_report()
        return self.history

    def _print_round_summary(self, round_data: Dict):
        print(f"\n轮次 {round_data['round']} 结果：")
        print(f"平均值：{round_data['average']:.2f}")
        print(f"目标值：{round_data['target']:.2f}")
        print("详细猜测：")
        for name, data in round_data["guesses"].items():
            status = "👑" if name == round_data["winner"] else "  "
            guess = f"{data['parsed_value']:.2f}" if data['parsed_value'] is not None else "无效"
            print(f"{status} {name}: {guess}")

    def _print_final_report(self):
        print("\n=== 终局报告 ===")
        print(f"总轮次：{len(self.history)}")

        # 统计胜率
        win_counts = {}
        for rd in self.history:
            if rd["winner"]:
                win_counts[rd["winner"]] = win_counts.get(rd["winner"], 0) + 1

        if win_counts:
            print("\n模型胜率统计：")
            for model in self.models:
                count = win_counts.get(model.model_name, 0)
                print(f"- {model.model_name}: {count}次 ({count / len(self.history):.1%})")

            best_model = max(win_counts, key=win_counts.get)
            print(f"\n最佳策略模型：{best_model}")
        else:
            print("没有有效优胜记录")

        # 计算策略有效性
        avg_diff = np.mean([abs(rd["target"] - rd["average"]) for rd in self.history])
        print(f"\n平均目标差值：{avg_diff:.2f}")


'''
if __name__ == "__main__":
    # 测试用例
    from models.mock_model import MockModel  # 假设存在模拟模型

    # 创建测试模型
    models = [
        MockModel("模型A", pattern=lambda r: str(30 + r * 5)),
        MockModel("模型B", pattern=lambda r: f"大约 {45 - r * 2} 左右"),
        MockModel("模型C", pattern=lambda r: f"[{50 + r * 3}]"),
    ]

    game = AdaptiveGuessGame(
        models=models,
        max_rounds=3,
        min_value=0,
        max_value=100
    )

    history = game.run_game()
'''