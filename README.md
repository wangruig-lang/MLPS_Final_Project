# MLPS Final Project - Power Outage Forecasting

CMU 95-828 Machine Learning for Problem Solving - Spring 2026

## 项目概述

本项目分为两个部分：
1. **Part I: 短期停电预测** — 用历史停电数据 + 天气数据预测 Michigan 83 个县未来 24h/48h 的每小时停电数（评分占 30%）
2. **Part II: 备用发电机摆放** — 基于 Part I 的预测结果，决定 5 台备用发电机如何分配到各县以最大化停电缓解（评分占 30%，含 report 论述）

## 当前进展

### Phase 1 已完成 (`phase1_eda.ipynb`)

| 任务 | 内容 | 关键方法 |
|------|------|---------|
| **Task A: EDA** | 停电数据探索 | GMM 数据驱动分层 (k=5)、地理热力图、风暴/日常分层分析 |
| **Task B: 特征筛选** | 天气特征筛选 | 4 方法共识投票 (Pearson/Spearman/Storm/RF)、共线性去冗余 (r>0.85)、符号一致性检查 |
| **Task C: 特征工程** | 通用特征构造 | 9 组 193 个特征 (时间/滞后/滚动/GMM分层/天气/交互/风暴指示) |
| **Task D: Baseline** | 模型评估 | Persistence、Historical Average、HistGBM、SARIMAX、Seq2Seq |

### 模型 RMSE 排行榜

| 模型 | 24h RMSE | 48h RMSE |
|------|----------|----------|
| SARIMAX | 27.91 | 20.13 |
| **HistGBM + 特征工程** | **54.37** | **44.02** |
| Historical Average | 93.03 | 73.55 |
| Seq2Seq (1-layer LSTM) | 100.69 | 109.38 |
| Persistence | 129.17 | 117.76 |

### 关键发现

- **停电数据特点**: 70.5% 为零值，极度右偏，极端值达 23346
- **GMM 自动分层**: 5 个regime (静默/轻微/日常/中等/严重/极端)，阈值 [3, 14, 77, 404]
- **特征重要性**: 停电滞后特征 (60%) >> 停电滚动窗口 (30%) >> 天气滚动 (6%) >> 风暴指示 (3%)
- **HistGBM 在 53% 的县优于 Persistence baseline**，但极端停电县 (26125, 26163) RMSE 极高

## 项目结构

```
├── phase1_eda.ipynb              # Phase 1: EDA + 特征工程 + Baseline 评估
├── demo.ipynb                    # 原始 demo（SARIMAX + Seq2Seq + Part II 发电机分配）
├── requirements.txt              # Python 依赖列表
├── .env.example                  # 环境变量模板
├── data/
│   ├── train.nc                  # 训练数据（~152MB, 2161h × 83 counties × 109 weather features）
│   ├── test_24h_demo.nc          # 24h 测试 demo 数据
│   └── test_48h_demo.nc          # 48h 测试 demo 数据
├── results/                      # 模型输出
│   ├── sarimax_pred_24h.csv
│   ├── sarimax_pred_48h.csv
│   ├── seq2seq_pred_24h.csv
│   └── seq2seq_pred_48h.csv
└── wandb/                        # W&B 本地日志（不追踪）
```

## Phase 1 Notebook 结构 (`phase1_eda.ipynb`)

### Task A: 停电数据探索
- **A1**: 基本统计 + 分布可视化（log-scale 直方图、箱线图）
- **A2**: GMM 数据驱动分层 — BIC 选 k，自动学习 regime 阈值，替代主观 P75/P99
- **A3**: 地理分析 — 时间×县热力图 (log1p)、plotly choropleth 地图
- **A4**: 分层时序分析 — P95 风暴检测 + 6h buffer、日常去零分析、bootstrap CI

### Task B: 天气特征筛选
- **B1**: 多方法相关性 — Pearson、Spearman (抽样 200k)、Storm Pearson、Quiet Nonzero Pearson
- **B2**: 贪心去冗余 — 按 Spearman 排序，逐步剔除 r>0.85 的冗余特征
- **B3**: 风暴解剖 — Top 4 风暴事件 ±24h 放大、天气 z-score 叠加、交叉相关 lag 分析
- **B4**: 缺失值检查
- **B5**: 共识投票 — 4 方法各选 Top-25 投票 + 符号一致性检查 → `final_selected`

### Task C: 通用特征构造
- **C1**: `build_features()` 封装 — 9 组特征自动构造 + NaN/泄漏检查
- **C2-C5**: 特征质量可视化、风暴行为、跨组共线性、工程总结

### Task D: Baseline 评估
- **D1**: 统一评估函数 `evaluate_model()` + `compare_models()`
- **D2**: Persistence + Historical Average baseline
- **D3**: HistGBM (LightGBM fallback) + 特征工程 + 特征重要性分析
- **D4**: 全模型对比可视化

## 后续计划 (Phase 2)

### Seq2Seq 改进方向
当前 Seq2Seq (1-layer LSTM, hidden=64, 5 epochs) 表现极差的原因：
- 模型太浅，未充分训练
- 未使用 Phase 1 的特征工程成果
- 未处理停电数据的极端稀疏性

改进路线：
1. **加深网络** — 2-3 层 LSTM/GRU，hidden_dim=128-256
2. **接入特征工程** — 使用 `build_features()` 的 193 维输入
3. **训练策略** — 更多 epochs (50-100)、学习率调度、早停
4. **损失函数** — 加权 MSE 或 Huber loss，对极端值加大惩罚

### 空间模型接入
- **Graph Neural Network (GNN)** — 以 83 县为节点，地理邻接为边，学习空间传播模式
- **建议**: 先把 Seq2Seq 调到合理水平 (RMSE < 50)，再叠加空间信息
- **原因**: 在弱 backbone 上加空间模块效果有限，且难以判断改进来自哪里

### 推荐训练顺序
1. 先用 HistGBM 做特征重要性参考 ✅ (已完成)
2. 改进 Seq2Seq → 接入特征工程 → 调参至合理水平
3. 加入 GNN 空间模块 → Seq2Seq + GNN 联合训练
4. 集成: HistGBM + Seq2Seq-GNN → 加权融合

## 环境配置

Notebook 支持 **本地运行** 和 **Google Colab 运行** 两种方式。

### 本地运行

```bash
git clone https://github.com/wangruig-lang/MLPS_Final_Project.git
cd MLPS_Final_Project
pip install -r requirements.txt
```

从 Canvas 下载 `train.nc` 放入 `data/` 目录。

### Google Colab

👉 [在 Colab 中打开 demo.ipynb](https://colab.research.google.com/github/wangruig-lang/MLPS_Final_Project/blob/main/demo.ipynb)

Runtime → Change runtime type → **T4 GPU** → Save，然后按顺序运行。

### Wandb 配置 (可选)

```bash
cp .env.example .env
# 编辑 .env 填入 WANDB_USERNAME, WANDB_API_KEY, WANDB_ENTITY
```

不配置 wandb 不影响 notebook 运行。

## Deadline

April 26th, 2026 at 11:59 PM
