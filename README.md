# AI游戏模拟项目

这个项目模拟了不同AI模型在各种博弈情境中的行为和决策过程。

## 游戏类型

- 猜数游戏 (guess)
- 囚徒困境 (prisoners)
- 市场模拟 (market)

## 如何使用

1. 配置API密钥：在`config.py`中配置各AI模型的API密钥
2. 运行主程序：`python main.py`
3. 修改`main.py`中的函数调用以运行不同的游戏

## 项目结构

- `main.py`: 主程序入口
- `config.py`: API密钥配置
- `run_games/`: 游戏运行逻辑
- `models/`: AI模型接口
- `games/`: 游戏定义
- `utils/`: 工具函数 