import numpy as np
from typing import List, Dict
from datetime import datetime
from utils.response_parser import extract_number
from models.base_model import BaseModel


class Guess70PercentGame:
    def __init__(
            self,
            models: List[BaseModel],
            max_rounds: int = 3,
            min_value: float = 0,
            max_value: float = 100
    ):
        """
        猜70%中值游戏控制器

        参数:
            models: 参与游戏的模型列表
            max_rounds: 最大游戏轮数（默认3轮）
            min_value: 数值范围下限
            max_value: 数值范围上限
        """
        self.models = models
        self.max_rounds = max_rounds
        self.min_value = min_value
        self.max_value = max_value
        self.results = []
        self.start_time = datetime.now()

    def _calculate_target(self, guesses: List[float]) -> float:
        """计算目标值（有效猜测的平均值*0.7）"""
        valid_guesses = [g for g in guesses if g is not None]
        return np.mean(valid_guesses) * 0.7 if valid_guesses else 0

    def _get_round_prompt(self, round_number: int) -> str:
        """生成本轮提示语"""
        return f"""
        请猜测所有参与者本轮猜测数字平均值的70%（{self.min_value}-{self.max_value}之间的数），当前是第{round_number}轮。
        请严格遵守回答格式：先说明思路，无论思路如何必须在最后给出一个确切的数字答案，例如：我认为...所以我给出的最终答案是...
        """

    def _determine_winner(self, guesses: Dict[str, float], target: float) -> str:
        """确定本轮获胜者"""
        closest = None
        min_diff = float('inf')

        for name, guess in guesses.items():
            if guess is None:
                continue

            # 处理超出范围的值
            clamped_guess = max(self.min_value, min(self.max_value, guess))
            diff = abs(clamped_guess - target)

            if diff < min_diff or (diff == min_diff and not closest):
                min_diff = diff
                closest = name

        return closest

    def play_round(self, round_number: int) -> Dict:
        """执行单轮游戏"""
        round_result = {
            "round": round_number,
            "guesses": {},
            "target": None,
            "winner": None,
            "timestamp": datetime.now().isoformat()
        }

        # 收集所有模型的猜测
        guesses = {}
        for model in self.models:
            try:
                prompt = self._get_round_prompt(round_number)
                response = model.generate_response(prompt)
                guess = extract_number(response)

                # 记录原始响应和解析结果
                round_result["guesses"][model.model_name] = {
                    "raw_response": response,
                    "parsed_value": guess
                }
                guesses[model.model_name] = guess
            except Exception as e:
                print(f"{model.model_name} 猜测失败: {str(e)}")
                guesses[model.model_name] = None

        # 计算目标值和胜者
        target = self._calculate_target(list(guesses.values()))
        winner = self._determine_winner(guesses, target)

        round_result.update({
            "target": round(target, 2),
            "winner": winner
        })

        self.results.append(round_result)
        return round_result

    def _print_round_summary(self, round_result: Dict):
        """打印本轮结果摘要"""
        print(f"\n=== 第 {round_result['round']} 轮结果 ===")
        print(f"目标值: {round_result['target']:.2f}")

        for model_name, data in round_result['guesses'].items():
            status = "✔" if model_name == round_result['winner'] else "✘"
            guess = data['parsed_value'] or "无效响应"
            print(f"{model_name}: {guess} {status}")

    def run_game(self):
        """运行完整游戏"""
        print(f"游戏开始，共进行{self.max_rounds}轮\n")

        for current_round in range(1, self.max_rounds + 1):
            print(f"\n第 {current_round} 轮进行中...")
            round_result = self.play_round(current_round)
            self._print_round_summary(round_result)

        self._print_final_report()
        return self.results

    def _print_final_report(self):
        """生成最终报告"""
        print("\n=== 游戏最终报告 ===")
        print(f"总耗时: {datetime.now() - self.start_time}")
        print(f"总轮数: {len(self.results)}")

        # 统计胜率
        win_counts = {}
        for round_data in self.results:
            winner = round_data["winner"]
            if winner:
                win_counts[winner] = win_counts.get(winner, 0) + 1

        print("\n胜率统计:")
        for model, count in win_counts.items():
            print(f"{model}: {count}胜（{count / len(self.results):.1%}）")

        # 展示最佳猜测
        best_round = min(
            self.results,
            key=lambda x: abs(
                x['target'] - np.mean([v['parsed_value'] for v in x['guesses'].values() if v['parsed_value']])))
        print(f"\n最接近轮次: 第{best_round['round']}轮（误差 {abs(best_round['target'] - np.mean([v['parsed_value'] for v in best_round['guesses'].values() if v['parsed_value']])):.2f})")
