from typing import List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from .prisoners_dilemma import PrisonersDilemmaGame, BaseModel, HumanPlayer

class ParallelPrisonersDilemma:
    """并行运行多对囚徒困境博弈的管理类"""
    
    def __init__(
        self,
        player_pairs: List[List[Union[BaseModel, HumanPlayer]]],
        rounds: int = 5,
        communication_start_round: Union[bool, int] = False,
        prompt_type: str = "normal",
        save_results: bool = True,
        results_dir: str = "game_results"
    ):
        """
        初始化并行博弈管理器
        
        Args:
            player_pairs: 玩家对列表，每个元素是一个包含两个玩家的列表
            rounds: 每对玩家进行博弈的轮数
            communication_start_round: 开始允许沟通的回合（False表示不允许沟通）
            prompt_type: 提示词类型 ("normal", "single", "opposite", "abstract")
            save_results: 是否保存博弈结果到文件
            results_dir: 结果保存的目录
        """
        self.player_pairs = player_pairs
        self.rounds = rounds
        self.communication_start_round = communication_start_round
        self.prompt_type = prompt_type
        self.save_results = save_results
        self.results_dir = results_dir
        self.games = []
        self.results = {}
        
        # 验证玩家对
        self._validate_player_pairs()
        
        # 初始化每对玩家的游戏实例
        self._initialize_games()
        
        if save_results and not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def _validate_player_pairs(self):
        """验证玩家对的有效性"""
        if not self.player_pairs:
            raise ValueError("至少需要一对玩家")
            
        for i, pair in enumerate(self.player_pairs):
            if len(pair) != 2:
                raise ValueError(f"玩家对 {i+1} 必须包含两个玩家")
            
            # 检查人类玩家数量
            human_count = sum(isinstance(p, HumanPlayer) for p in pair)
            if human_count > 1:
                raise ValueError(f"玩家对 {i+1} 最多只能包含一个人类玩家")
    
    def _initialize_games(self):
        """为每对玩家初始化游戏实例"""
        for pair in self.player_pairs:
            game = PrisonersDilemmaGame(
                players=pair,
                rounds=self.rounds,
                communication_start_round=self.communication_start_round,
                prompt_type=self.prompt_type,
                wait_for_input=False  # 禁用单个游戏的等待输入
            )
            self.games.append(game)

    def _process_round_data(self, round_data: Dict) -> Dict:
        """处理单轮数据，提取关键信息"""
        processed_data = {
            "round": round_data["round"],
            "decisions": {},
            "scores": round_data["scores"]
        }
        
        # 提取每个玩家的关键决策信息
        for player, data in round_data["decisions"].items():
            thought = data.get("thought", "")
            decision = data.get("decision", "")
            
            # 检查是否有效的决策数据
            if not thought or not decision:
                thought = "【未提供有效的策略分析】"
                
            processed_data["decisions"][player] = {
                "thought": thought,
                "decision": decision if decision else "未决策",
                "is_valid": bool(thought and decision)
            }
        
        # 如果有沟通消息，保留它
        if round_data.get("messages"):
            processed_data["messages"] = round_data["messages"]
            
        return processed_data
    
    def _run_single_game(self, game_index: int) -> Dict:
        """运行单个博弈实例并返回简化的结果"""
        game = self.games[game_index]
        pair = self.player_pairs[game_index]
        
        print(f"\n{'='*50}")
        print(f"开始第 {game_index + 1} 组博弈")
        print(f"参与者: {pair[0].model_name} vs {pair[1].model_name}")
        print(f"{'='*50}\n")
        
        history = game.run_game()
        
        # 计算总分
        final_scores = {
            pair[0].model_name: sum(r["scores"][pair[0].model_name] for r in history),
            pair[1].model_name: sum(r["scores"][pair[1].model_name] for r in history)
        }
        
        # 计算合作率
        cooperation_rates = {}
        for p in pair:
            coop_count = sum(1 for r in history if r["decisions"][p.model_name]["decision"] == "合作")
            cooperation_rates[p.model_name] = round(coop_count / len(history) * 100, 2)
        
        # 构建简化的结果数据
        result = {
            "game_index": game_index,
            "players": [p.model_name for p in pair],
            "rounds_data": [self._process_round_data(round_data) for round_data in history],
            "summary": {
                "final_scores": final_scores,
                "cooperation_rates": cooperation_rates,
                "total_rounds": len(history)
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def _generate_overall_summary(self, results: List[Dict]) -> Dict:
        """生成所有博弈的总体汇总数据"""
        all_players = set()
        player_stats = {}
        
        # 收集所有玩家
        for result in results:
            all_players.update(result["players"])
        
        # 初始化玩家统计
        for player in all_players:
            player_stats[player] = {
                "total_score": 0,
                "games_played": 0,
                "avg_cooperation_rate": 0.0,
                "wins": 0
            }
        
        # 统计数据
        for result in results:
            scores = result["summary"]["final_scores"]
            coop_rates = result["summary"]["cooperation_rates"]
            
            # 找出这场游戏的赢家
            winner = max(scores.items(), key=lambda x: x[1])[0]
            
            for player in result["players"]:
                stats = player_stats[player]
                stats["total_score"] += scores[player]
                stats["games_played"] += 1
                stats["avg_cooperation_rate"] += coop_rates[player]
                if player == winner:
                    stats["wins"] += 1
        
        # 计算平均值
        for stats in player_stats.values():
            if stats["games_played"] > 0:
                stats["avg_cooperation_rate"] = round(
                    stats["avg_cooperation_rate"] / stats["games_played"], 
                    2
                )
        
        return {
            "player_statistics": player_stats,
            "total_games": len(results)
        }
    
    def _generate_markdown_report(self, results: List[Dict], overall_summary: Dict) -> str:
        """生成markdown格式的实验报告"""
        report = []
        
        # 标题和实验概况
        report.extend([
            "# 囚徒困境博弈实验报告\n",
            "## 实验概况",
            f"- 总博弈组数：{len(results)}",
            f"- 每组回合数：{self.rounds}",
            f"- 提示词类型：{self.prompt_type}",
            f"- 沟通机制：{'开启' if self.communication_start_round is not False else '关闭'}"
        ])
        
        if self.communication_start_round not in [False, 0]:
            report.append(f"  - 开始回合：第 {self.communication_start_round + 1} 回合")
        
        # 各组博弈详情
        report.extend([
            "\n## 各组博弈详情"
        ])
        
        for result in sorted(results, key=lambda x: x["game_index"]):
            report.extend([
                f"\n### 第 {result['game_index'] + 1} 组",
                "#### 对阵情况",
                "| 参与者 | 最终得分 | 合作率 |",
                "|--------|----------|--------|"
            ])
            
            for player in result["players"]:
                score = result["summary"]["final_scores"][player]
                coop_rate = result["summary"]["cooperation_rates"][player]
                report.append(f"| {player} | {score} | {coop_rate}% |")
                
            # 添加每轮详细数据
            report.extend([
                "\n#### 回合详情"
            ])
            
            for round_data in result["rounds_data"]:
                round_num = round_data["round"]
                report.append(f"\n##### 第 {round_num} 回合")
                
                # 添加决策表格
                report.extend([
                    "| 玩家 | 决策 | 得分 |",
                    "|------|------|------|"
                ])
                
                for player in result["players"]:
                    decision = round_data["decisions"][player]
                    score = round_data["scores"][player]
                    is_valid = decision.get("is_valid", True)
                    decision_text = decision["decision"]
                    if not is_valid:
                        decision_text += " (无效数据)"
                    
                    report.append(f"| {player} | {decision_text} | {score} |")
                
                # 添加思考过程（完整版）
                report.append("\n**思考过程：**")
                
                for player in result["players"]:
                    decision = round_data["decisions"][player]
                    thought = decision["thought"].replace("\n", "<br>").replace("|", "\\|")
                    report.extend([
                        f"\n*{player}*：",
                        f"{thought}"
                    ])
                    
                # 如果有沟通消息，添加到回合详情中
                if round_data.get("messages"):
                    report.extend([
                        "\n**沟通消息：**",
                        "| 玩家 | 消息 |",
                        "|------|------|"
                    ])
                    for player, message in round_data["messages"].items():
                        report.append(f"| {player} | {message} |")
        
        # 总体统计
        report.extend([
            "\n## 总体统计",
            "\n### 玩家表现排名"
        ])
        
        # 按总得分排序
        sorted_players = sorted(
            overall_summary["player_statistics"].items(),
            key=lambda x: (x[1]["total_score"], x[1]["wins"]),
            reverse=True
        )
        
        report.extend([
            "\n| 排名 | 玩家 | 总得分 | 参与场次 | 胜利场次 | 平均合作率 |",
            "|------|------|--------|----------|----------|------------|"
        ])
        
        for rank, (player, stats) in enumerate(sorted_players, 1):
            report.append(
                f"| {rank} | {player} | {stats['total_score']} | "
                f"{stats['games_played']} | {stats['wins']} | "
                f"{stats['avg_cooperation_rate']}% |"
            )
        
        # 策略分析
        report.extend([
            "\n### 策略分析",
            "\n各玩家倾向："
        ])
        
        for player, stats in sorted_players:
            coop_rate = stats["avg_cooperation_rate"]
            tendency = "高度合作型" if coop_rate >= 80 else \
                      "倾向合作型" if coop_rate >= 60 else \
                      "混合策略型" if coop_rate >= 40 else \
                      "倾向背叛型" if coop_rate >= 20 else \
                      "高度背叛型"
            win_rate = (stats["wins"] / stats["games_played"]) * 100
            avg_score = stats['total_score']/stats['games_played']
            
            report.extend([
                f"\n- **{player}**",
                f"  - 策略倾向：{tendency}",
                f"  - 胜率：{win_rate:.1f}%",
                f"  - 平均每场得分：{avg_score:.1f}"
            ])
        
        return "\n".join(report)

    def run_all_games(self, max_parallel: Optional[int] = None) -> Dict:
        """
        并行运行所有博弈
        
        Args:
            max_parallel: 最大并行运行的游戏数，None表示无限制
        
        Returns:
            包含所有博弈结果的字典
        """
        results = []
        
        # 如果有人类玩家，则串行运行
        if any(isinstance(p, HumanPlayer) for pair in self.player_pairs for p in pair):
            print("检测到人类玩家，将按顺序运行博弈...")
            for i in range(len(self.games)):
                results.append(self._run_single_game(i))
        else:
            # 使用线程池并行运行AI之间的博弈
            max_workers = min(len(self.games), max_parallel or len(self.games))
            print(f"并行运行 {len(self.games)} 组博弈（最大并行数：{max_workers}）...")
            
            # 按回合进行并行处理
            for current_round in range(1, self.rounds + 1):
                print(f"\n{'='*20} 第 {current_round} 回合 {'='*20}")
                
                # 并行运行当前回合
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 为每个游戏创建一个任务
                    futures = []
                    for i, game in enumerate(self.games):
                        if len(results) <= i:
                            # 第一轮，初始化结果
                            results.append({
                                "game_index": i,
                                "players": [p.model_name for p in self.player_pairs[i]],
                                "rounds_data": [],
                                "summary": {
                                    "final_scores": {p.model_name: 0 for p in self.player_pairs[i]},
                                    "cooperation_rates": {p.model_name: 0 for p in self.player_pairs[i]},
                                    "total_rounds": 0
                                }
                            })
                        
                        # 提交任务
                        future = executor.submit(game.play_round, current_round)
                        futures.append((i, future))
                    
                    # 等待所有任务完成并收集结果
                    for i, future in futures:
                        round_data = future.result()
                        results[i]["rounds_data"].append(self._process_round_data(round_data))
                        
                        # 更新汇总信息
                        for player in results[i]["players"]:
                            results[i]["summary"]["final_scores"][player] += round_data["scores"][player]
                
                # 打印当前回合所有游戏的结果
                print("\n当前回合结果汇总：")
                for result in results:
                    print(f"\n第 {result['game_index'] + 1} 组：")
                    for player in result["players"]:
                        score = result["summary"]["final_scores"][player]
                        coop_count = sum(1 for r in result["rounds_data"] 
                                       if r["decisions"][player]["decision"] == "合作")
                        coop_rate = round(coop_count / len(result["rounds_data"]) * 100, 2)
                        result["summary"]["cooperation_rates"][player] = coop_rate
                        
                        # 检查是否有无效数据
                        invalid_rounds = sum(1 for r in result["rounds_data"] 
                                          if not r["decisions"][player].get("is_valid", True))
                        if invalid_rounds > 0:
                            print(f"  {player}: 累计得分 {score}, 合作率 {coop_rate}% (有 {invalid_rounds} 轮无效数据)")
                        else:
                            print(f"  {player}: 累计得分 {score}, 合作率 {coop_rate}%")
                
                # 如果不是最后一轮，等待用户确认
                if current_round < self.rounds:
                    user_input = input("\n所有对局当前回合已完成。按回车继续下一回合，输入q退出: ")
                    if user_input.lower() == 'q':
                        print("实验提前终止")
                        break
        
        # 生成总体汇总数据
        summary = self._generate_overall_summary(results)
        
        # 生成markdown报告
        markdown_report = self._generate_markdown_report(results, summary)
        
        # 保存结果
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存Markdown格式的报告
            md_filename = os.path.join(self.results_dir, f"game_report_{timestamp}.md")
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(markdown_report)
            
            print(f"\n实验报告已保存至：`{md_filename}`")
        
        # 打印汇总结果
        print(markdown_report)
        
        return results 