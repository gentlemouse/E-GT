import re
import random
from typing import Optional
def extract_number(text: str) -> Optional[float]:
    """
    增强版数字提取函数（提取最后一个有效数字）
    支持格式：
    - 整数（42）
    - 小数（3.14）
    - 带符号数字（-5.6 / +7.8）
    """
    pattern = r"[-+]?\d+\.\d+|[-+]?\d+"
    matches = re.findall(pattern, text)

    if matches:
        try:
            # 取最后一个匹配项并转换为float
            return float(matches[-1])
        except (ValueError, TypeError):
            pass
    return None


def parse_decision(text: str) -> str:
    text = text.lower().strip()

    # 精确匹配
    if "enter" in text or "yes" in text:
        return "进入"
    if "exit" in text or "no" in text:
        return "不进入"

    # 正则匹配
    patterns = {
        "进入": r"(?i)(参与|加入|entry|join)",
        "不进入": r"(?i)(退出|放弃|exit|opt\s*out)"
    }

    for decision, pattern in patterns.items():
        if re.search(pattern, text):
            return decision

    # 基于概率的预测
    positive_keywords = ["有利可图", "需求大", "机会"]
    negative_keywords = ["饱和", "风险", "竞争激烈"]

    pos_score = sum(1 for kw in positive_keywords if kw in text)
    neg_score = sum(1 for kw in negative_keywords if kw in text)

    return "进入" if pos_score > neg_score else "不进入"


def parse_cooperation_decision(text: str) -> str:
    """解析合作/背叛决策"""
    text = text.lower().replace(" ", "")

    # 精确匹配
    if "合作" in text or "cooperate" in text or "c" == text:
        return "合作"
    if "背叛" in text or "defect" in text or "d" == text:
        return "背叛"

    # 模糊匹配
    patterns = {
        "合作": [r"(?i)(合作|协作|一起|共赢|trust)"],
        "背叛": [r"(?i)(背叛|欺骗|自私|背刺|cheat)"]
    }

    for decision, regex_list in patterns.items():
        for pattern in regex_list:
            if re.search(pattern, text):
                return decision

    # 默认随机选择（避免无效决策）
    return random.choice(["合作", "背叛"])