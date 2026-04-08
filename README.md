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
| **Task C: 特征工程 v2** | 精简特征构造 | 193维→Tier A(~30维)/Tier B(~42维)、PCA天气压缩、负贡献组移除 |
| **Task D: Baseline** | 模型评估 | Persistence、Historical Average、HistGBM (Tier A/B 对比)、SARIMAX、Seq2Seq |

### 特征工程 v2 设计

基于 HistGBM 特征重要性分析，将 193 维特征精简为两个层级：

| 特征组 | 重要性 | v1 维度 | v2 维度 | 决策 |
|--------|--------|---------|---------|------|
| outage_lag | 60.4% | 10 | 10 | 保留 |
| outage_rolling | 30.2% | 15 | 15 | 保留 |
| storm_indicator | 3.1% | 3 | 3 | 保留 |
| outage_regime | 1.0% | 2 | 2 | 保留 (Tier B) |
| weather_rolling | 6.3% | 42 | ≤12 | 精简 (仅24h) |
| weather_raw | 0.5% | 88 | 3 (PCA) | PCA 压缩 |
| time | 0.0% | 9 | 4 | 精简 (仅sin/cos) |
| weather_lag | -0.8% | 18 | 0 | 移除 |
| weather_interaction | -0.8% | 6 | 0 | 移除 |

- **Tier A (compact, ~30 维, ~94% 重要性)**: 适合 LSTM/Seq2Seq/GRU
- **Tier B (full, ~42 维, ~99% 重要性)**: 适合 Transformer/GBM/Ensemble

### 模型 RMSE 排行榜

| 模型 | 24h RMSE | 48h RMSE |
|------|----------|----------|
| SARIMAX | 27.91 | 20.13 |
| **HistGBM + Tier B** | **54.37** | **44.02** |
| Historical Average | 93.03 | 73.55 |
| Seq2Seq (1-layer LSTM) | 100.69 | 109.38 |
| Persistence | 129.17 | 117.76 |

### 关键发现

- **停电数据特点**: 70.5% 为零值，极度右偏，极端值达 23346
- **GMM 自动分层**: 5 个regime (静默/轻微/日常/中等/严重/极端)，阈值 [3, 14, 77, 404]
- **特征重要性**: 停电滞后特征 (60%) >> 停电滚动窗口 (30%) >> 天气滚动 (6%) >> 风暴指示 (3%)
- **v1 → v2 精简**: 193 维中 112 维 (58%) 贡献 ≤ 0%，PCA 将 88 维天气压缩至 3 维

## 项目结构

```
├── phase1_eda.ipynb              # Phase 1: EDA + 特征工程 v2 + Baseline 评估
├── model_template.ipynb          # 通用模型训练模板 (Config/DataLoader/Model/Eval)
├── model_deep_lstm.ipynb         # Track A: Deep LSTM + GNN 实现
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

## 特征工程 v2 调用方式

```python
# 在后续 notebook 中调用:
df, tier_compact, tier_full, pca_info = build_features(
    ds,
    selected_weather_features=selected_features,
    top_weather_features=final_selected[:6],
    gmm_thresholds=gmm_thresholds,
)

# 选择特征层级
feature_cols = tier_compact  # LSTM/Seq2Seq/GRU → ~30 维, ~94% 重要性
feature_cols = tier_full     # Transformer/GBM  → ~42 维, ~99% 重要性

# 构造训练数据
X = df[feature_cols].values
y = df['out'].values
```

### Test Set PCA 转换

```python
# 对 test set 天气数据应用训练集的 PCA:
w_test_normalized = (w_test_selected - pca_info['w_mean']) / pca_info['w_std']
w_test_pca = pca_info['pca_model'].transform(w_test_normalized)
```

### 返回值说明

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `df` | DataFrame | 每行 = (timestamp, location)，包含所有特征列 |
| `tier_compact` | list[str] | Tier A 特征列名 (~30 维) |
| `tier_full` | list[str] | Tier B 特征列名 (~42 维，包含 Tier A 全部) |
| `pca_info` | dict | PCA 模型、解释方差、标准化参数 |

## Phase 2: 模型训练分工

### Track A: LSTM → GNN (`model_deep_lstm.ipynb`)
1. Deep LSTM (3层, hidden=256, residual + LayerNorm)
2. SpatialWrapper: LSTM + GCN 空间传播 + 门控融合

### Track B: Seq2Seq → Two-Stage
1. Seq2Seq + Attention (BahdanauAttention)
2. Two-Stage: 风暴分类器 → 条件预测

### Track C: Transformer → Ensemble
1. TransformerForecaster (多头自注意力)
2. EnsemblePredictor: 加权融合所有模型

## 环境配置

### 本地运行

```bash
git clone https://github.com/wangruig-lang/MLPS_Final_Project.git
cd MLPS_Final_Project
pip install -r requirements.txt
```

从 Canvas 下载 `train.nc` 放入 `data/` 目录。

### Google Colab

Runtime → Change runtime type → **T4 GPU** → Save，然后按顺序运行。

### Wandb 配置 (可选)

```bash
cp .env.example .env
# 编辑 .env 填入 WANDB_USERNAME, WANDB_API_KEY, WANDB_ENTITY
```

不配置 wandb 不影响 notebook 运行。

## Deadline

April 26th, 2026 at 11:59 PM
