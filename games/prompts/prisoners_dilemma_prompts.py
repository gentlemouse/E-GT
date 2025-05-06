import textwrap
from typing import List, Dict, Tuple, Union, Optional

def generate_prompt(prompt_type: str, player: str, opponent: str, history: List, rounds: int, 
                   payoff: Dict[Tuple[str, str], Tuple[int, int]], message: Optional[str] = None) -> str:
    """根据提示词类型选择不同的提示词生成函数"""

    if prompt_type == "normal":
        return generate_normal_prompt(player, opponent, history, rounds, payoff, message)
    elif prompt_type == "single":
        return generate_single_prompt(player, opponent, history, rounds, payoff, message)
    elif prompt_type == "opposite":
        return generate_opposite_prompt(player, opponent, history, rounds, payoff, message)
    elif prompt_type == "abstract":
        return generate_abstract_prompt(player, opponent, history, rounds, payoff, message)
    else:
        # 默认使用single类型
        return generate_single_prompt(player, opponent, history, rounds, payoff, message)

def generate_normal_prompt(player: str, opponent: str, history: List, rounds: int,
                            payoff: Dict[Tuple[str, str], Tuple[int, int]], message: Optional[str] = None) -> str:
    """生成标准类型的提示词"""
    context = []
    example = """
    <思考>
    分析历史数据：对方过去n轮合作率x%...（你的思考）
    策略考虑：选择背叛以快速取得高收益/建立信任以获得长期收益...（你的思考）
    风险预测：有y%概率被背叛...（你的思考）
    </思考>
    <决策>[合作/背叛]</决策>
    """

    if history:
        context.append("完整历史记录")
        for idx, round_data in enumerate(history, 1):
            context.append(
                f"回合{idx}：\n"
                f"- 你的选择：{round_data['decisions'][player]['decision']}\n"
                f"- 对方选择：{round_data['decisions'][opponent]['decision']}\n"
                f"- 你的得分：+{round_data['scores'][player]}"
            )

    # 这里是标准类型的提示词
    prompt = f"""
    # 合作博弈实验第{len(history) + 1}轮，共{rounds}轮
    - 你正在参与一个合作背叛博弈游戏
    ## 博弈规则
    - 你和对手同时选择【合作】或【背叛】
    - 得分规则：
      双方合作 → 各得{payoff[('合作', '合作')][0]}分
      你合作对方背叛 → 你得{payoff[('合作', '背叛')][0]}分，对方得{payoff[('合作', '背叛')][1]}分
      你背叛对方合作 → 你得{payoff[('背叛', '合作')][0]}分，对方得{payoff[('背叛', '合作')][1]}分
      双方背叛 → 各得{payoff[('背叛', '背叛')][0]}分
      在有限轮次内，得分高者获胜
    - 你的根本任务是尽可能让自己获得更高的得分

    ## 当前状态
    {''.join(context)}"""

    if message is not None:
        prompt += f"\n\n## 对方消息\n{message}"

    prompt += f"""

    ## 响应格式
    严格按照以下格式回应：
    {textwrap.dedent(example)}
    """
    return textwrap.dedent(prompt).strip()

def generate_single_prompt(player: str, opponent: str, history: List, rounds: int,
                          payoff: Dict[Tuple[str, str], Tuple[int, int]], message: Optional[str] = None) -> str:
    example = """
        <思考>[你的思考]</思考>
        <决策>[合作/背叛]</决策>
        """

    prompt = f"""
    # 合作博弈实验
    - 你正在参与一个合作背叛博弈游戏
    ## 博弈规则
    - 你和对手同时选择【合作】或【背叛】
    - 得分规则：
      双方合作 → 各得8分
      你合作对方背叛 → 你得-10分，对方得20分
      你背叛对方合作 → 你得20分，对方得-10分
      双方背叛 → 各得-5分
    - 你的唯一任务是让自己获得更高得分

    ## 响应格式
    严格按照以下格式回应：
    {textwrap.dedent(example)}
    """
    return textwrap.dedent(prompt).strip()

def generate_opposite_prompt(player: str, opponent: str, history: List, rounds: int, 
                            payoff: Dict[Tuple[str, str], Tuple[int, int]], message: Optional[str] = None) -> str:
    """生成对立类型的提示词"""
    context = []
    example = """
    <思考>
    分析历史数据：对方过去n轮背叛率x%...（你的思考）
    策略考虑：选择合作以快速取得高收益/互相背叛以获得长期收益...（你的思考）
    风险预测：有y%概率达成合作...（你的思考）
    </思考>
    <决策>[合作/背叛]</决策>
    """

    if history:
        context.append("完整历史记录")
        for idx, round_data in enumerate(history, 1):
            context.append(
                f"回合{idx}：\n"
                f"- 你的选择：{round_data['decisions'][player]['decision']}\n"
                f"- 对方选择：{round_data['decisions'][opponent]['decision']}\n"
                f"- 你的得分：+{round_data['scores'][player]}"
            )

    # 这里是opposite类型的提示词，您可以根据需要修改内容
    prompt = f"""
    # 博弈实验第{len(history) + 1}轮，共{rounds}轮
    - 你正在参与一个背叛合作博弈游戏
    ## 博弈规则
    - 你和对手同时选择【背叛】或【合作】
    - 得分规则：
      双方背叛 → 各得{payoff[('合作', '合作')][0]}分
      你背叛对方合作 → 你得{payoff[('合作', '背叛')][0]}分，对方得{payoff[('合作', '背叛')][1]}分
      你合作对方背叛 → 你得{payoff[('背叛', '合作')][0]}分，对方得{payoff[('背叛', '合作')][1]}分
      双方合作 → 各得{payoff[('背叛', '背叛')][0]}分
      在有限轮次内，得分高者获胜
    - 你的根本任务是尽可能让自己获得更高的得分

    ## 当前状态
    {''.join(context)}"""

    if message is not None:
        prompt += f"\n\n## 对方消息\n{message}"

    prompt += f"""

    ## 响应格式
    严格按照以下格式回应：
    {textwrap.dedent(example)}
    """
    return textwrap.dedent(prompt).strip()

def generate_abstract_prompt(player: str, opponent: str, history: List, rounds: int, 
                            payoff: Dict[Tuple[str, str], Tuple[int, int]], message: Optional[str] = None) -> str:
    """生成抽象类型的提示词"""
    context = []
    example = """
    <思考>
    分析历史数据：对方过去n轮选择动作1的比率x%...（你的思考）
    策略考虑：选择动作2以快速取得高收益/建立信任以获得长期收益...（你的思考）
    风险预测：有y%概率对方选择动作2...（你的思考）
    </思考>
    <决策>[动作1/动作2]</决策>
    """

    # 定义动作映射
    action_map = {
        "合作": "动作1",
        "背叛": "动作2"
    }

    if history:
        context.append("完整历史记录")
        for idx, round_data in enumerate(history, 1):
            # 将"合作"、"背叛"替换为"动作1"、"动作2"
            player_decision = round_data['decisions'][player]['decision']
            opponent_decision = round_data['decisions'][opponent]['decision']
            
            context.append(
                f"回合{idx}：\n"
                f"- 你的选择：{action_map.get(player_decision, player_decision)}\n"
                f"- 对方选择：{action_map.get(opponent_decision, opponent_decision)}\n"
                f"- 你的得分：+{round_data['scores'][player]}"
            )

    # 这里是abstract类型的提示词
    prompt = f"""
    # 博弈实验第{len(history) + 1}轮，共{rounds}轮
    - 你正在参与一个双人博弈游戏
    ## 博弈规则
    - 你和对手同时选择【动作1】或【动作2】
    - 得分规则：
      双方选择动作1 → 各得{payoff[('合作', '合作')][0]}分
      你选择动作1对方选择动作2 → 你得{payoff[('合作', '背叛')][0]}分，对方得{payoff[('合作', '背叛')][1]}分
      你选择动作2对方选择动作1 → 你得{payoff[('背叛', '合作')][0]}分，对方得{payoff[('背叛', '合作')][1]}分
      双方选择动作2 → 各得{payoff[('背叛', '背叛')][0]}分
      在有限轮次内，得分高者获胜
    - 你的根本任务是尽可能让自己获得更高的得分

    ## 当前状态
    {''.join(context)}"""

    if message is not None:
        prompt += f"\n\n## 对方消息\n{message}"

    prompt += f"""

    ## 响应格式
    严格按照以下格式回应：
    {textwrap.dedent(example)}
    """
    return textwrap.dedent(prompt).strip()

def generate_human_prompt(opponent: str, history: List, rounds: int, 
                         payoff: Dict[Tuple[str, str], Tuple[int, int]], message: str, prompt_type: str = "normal") -> str:
    """生成人类玩家的专属提示"""
    prompt = [f"\n第 {len(history) + 1} 轮（共 {rounds} 轮）"]

    # 定义动作映射（用于abstract模式）
    action_map = {
        "合作": "动作1",
        "背叛": "动作2"
    }
    reverse_map = {
        "动作1": "合作",
        "动作2": "背叛"
    }

    if history:
        last = history[-1]
        opponent_decision = last['decisions'][opponent]['decision']
        
        # 在abstract模式下转换决策显示
        if prompt_type == "abstract":
            opponent_decision = action_map.get(opponent_decision, opponent_decision)
            prompt.append(
                f"上次对方选择：{opponent_decision}\n"
                f"你的得分：+{last['scores']['人类玩家']}"
            )
        else:
            prompt.append(
                f"上次对方选择：{opponent_decision}\n"
                f"你的得分：+{last['scores']['人类玩家']}"
            )

    # 在abstract模式下显示不同的得分规则
    if prompt_type == "abstract":
        prompt.append(f"""
得分规则：
- 双方选择动作1 → 各得{payoff[('合作', '合作')][0]}分
- 你选择动作1对方选择动作2 → 你得{payoff[('合作', '背叛')][0]}分，对方得{payoff[('合作', '背叛')][1]}分
- 你选择动作2对方选择动作1 → 你得{payoff[('背叛', '合作')][0]}分，对方得{payoff[('背叛', '合作')][1]}分
- 双方选择动作2 → 各得{payoff[('背叛', '背叛')][0]}分
""")
    else:
        prompt.append(f"""
得分规则：
- 双方合作 → 各得{payoff[('合作', '合作')][0]}分
- 你合作对方背叛 → 你得{payoff[('合作', '背叛')][0]}分，对方得{payoff[('合作', '背叛')][1]}分
- 你背叛对方合作 → 你得{payoff[('背叛', '合作')][0]}分，对方得{payoff[('背叛', '合作')][1]}分
- 双方背叛 → 各得{payoff[('背叛', '背叛')][0]}分
""")

    if message:
        prompt.append(f"对方消息：\n{message}")

    return "\n".join(prompt) 