# MLPS Final Project — 阶段性工作笔记

**更新日期**: 2026-04-20
**状态**: Phase 1 模型探索 → 切换主力到 HistGBM，准备优化
**Deadline**: 2026-04-26 23:59

---

## 0. TL;DR（如果你只想读一段）

- **当前最佳模型: HistGBM + Tier B 特征**，24h RMSE = **92.17**，48h = **80.81**（原始停电数空间，interleaved split）
- **DeepLSTM 一路从 v2 调到 v9 都跑不过 HistAvg 基线**（最好的 v7 是 113.47，等于 HistAvg），**确认是架构瓶颈不是特征瓶颈**
- 下一步：超参调优 HistGBM + Detroit 特殊加权 → 目标 <85；然后特征工程 v3 → 目标 <75
- LSTM 和 GNN 的结果全部保留（checkpoint + val pred 在 `results/`），report 里当"探索对比"

---

## 1. 项目设定回顾

- **Part I**: 用历史停电 + 天气数据预测 Michigan 83 个县未来 24h/48h 的每小时停电数
- **评估指标**: **县均 RMSE**（per-county RMSE 再取 mean），原始停电数空间
- **Part II**: 基于 Part I 预测结果分配 5 台备用发电机
- 数据: `data/train.nc`，2161h × 83 counties × 109 weather features

---

## 2. 数据与特征工程

### 2.1 特征工程 v2（phase1_eda.ipynb 里定稿，model_deep_lstm.ipynb §1.4 调用）

| 特征组 | v1 维度 | v2 维度 | 处理 |
|---|---|---|---|
| outage_lag | 10 | 10 | 保留 (lags=[1,3,6,12,24]) |
| outage_rolling | 15 | 15 | 保留 (wins=[6,12,24], mean/max/std/sum) |
| storm_indicator | 3 | 3 | 保留 (n_weather_anomaly / score / max_6h) |
| outage_regime | 2 | 2 | 保留 (Tier B) |
| weather_rolling | 42 | ≤12 | 精简，只留 top-6 天气的 24h rmean/rmax |
| weather_raw | 88 | 3 | **PCA 压缩到 3 维** |
| time | 9 | 4 | 精简 (只留 hour_sin/cos, dow_sin/cos) |
| weather_lag | 18 | 0 | 移除（贡献 -0.8%） |
| weather_interaction | 6 | 0 | 移除（贡献 -0.8%） |

- **Tier A** (compact, ~28 维, ~94% 重要性) — 建议 LSTM/Seq2Seq/GRU 用
- **Tier B** (full, ~42 维, ~99% 重要性) — 建议 Transformer/GBM/Ensemble 用

### 2.2 Train/Val Split（最关键的一个修复）

**老方案 (v2-v6)**: Chronological split, 前 80% 训练 + 后 20% 验证
- 问题: 6 月下旬一场大风暴**全部落在 val**，train 里完全没见过类似事件
- 症状: train 均值 32.7, val 均值 87.9 — **分布严重不匹配**
- LSTM val RMSE (log 空间) 卡在 1.38 下不去，以为是过拟合，其实是分布外问题

**新方案 (v7+)**: Interleaved split，每 5 天取第 1 天做 val
```python
stride = int(round(1.0 / VALIDATION_SPLIT))  # 5
is_val = (days_since_start % stride == 0)
```
- train 均值 46.7, val 均值 32.9 — 分布匹配
- val RMSE (log 空间) 从 1.38 降到 1.02，**证实之前的"过拟合"其实是分布不匹配**

**⚠ 这意味着老 baseline 数字无法直接比**：
- phase1_eda.ipynb 里的 HistGBM 24h=54.37 / 48h=44.02 是在老 chronological split 上
- 新 interleaved split 上的 HistGBM 24h=92.17 / 48h=80.81 才是可对比的真实数字

---

## 3. 模型实验日志（按时间顺序）

### 3.1 Baseline（新 interleaved split，原始停电数空间）

| 模型 | 24h | 48h | full | 备注 |
|---|---|---|---|---|
| Zero | 116.10 | 100.46 | 148.94 | 一律预测 0 |
| HistAvg | 113.42 | 102.00 | 148.36 | train 每县均值 |
| Persistence | 153.40 | 136.66 | 183.78 | train 最后时刻外推 |

### 3.2 DeepLSTM v6（老 split，失败）

- **Config**: chronological split + log1p target + weighted_mse
- **结果**: DeepLSTM_24h = 117.83 — **比 Zero=116 还差**
- **失败原因**: 分布不匹配 + log 空间 expm1 把小误差放大成大误差 + Detroit (26125) 单县 RMSE 压爆

### 3.3 DeepLSTM v7: rate target + Huber δ=0.01（interleaved split，躺平）

- **改动**:
  - `TARGET_TYPE = "rate"` (y = out / tracked)
  - `LOSS_FN = "huber"`, `HUBER_DELTA = 0.01`
  - Split 改 interleaved
- **训练日志**:
  - TrLoss epoch 1 = 0.0000 — **模型从第一步就不学了**
  - ValRMSE (rate space) = 0.0100 始终不降
  - 等价于一律预测 rate = 0
- **原始空间 RMSE**:
  - DeepLSTM_24h = **113.47** (跟 HistAvg 打平)
  - DeepLSTM_48h = 99.64 (比 Zero 好 0.8 点)
  - DeepLSTM_full = 147.30
  - **Pred 值域: [0, 1425]** vs 真值 [0, 11903] — **峰值只到真值 12%，严重压平**
- **GNN 叠加**:
  - DeepLSTM_GNN_24h = 113.04 (仅 **+0.4%**)
  - 81/83 县上 GNN 和 LSTM 几乎打平（都在预测 0）
- **Per-county 灾区 (24h)**:
  - 26125 (Detroit): **1807**
  - 26163 (Wayne): 956
  - 26161 (Washtenaw): 934
  - 26051 (Gladwin): 820
- **诊断**: rate 目标 + huber δ=0.01（在 rate 空间只有 1% 的惩罚门槛）+ 70% 样本是 0 → **模型发现"一律预测 0"就能把 huber loss 刷到 0**，根本没学。

### 3.4 DeepLSTM v8: rate + weighted_mse, w=20（过度矫正）

- **改动**: `LOSS_FN = "weighted_mse"`, `EXTREME_WEIGHT = 20.0`, `EXTREME_THRESHOLD = 0.005`
- **训练日志**:
  - TrLoss epoch 1 = 0.0397 → 最终 0.0011（**模型在学**了）
  - ValRMSE (rate) = 0.0116（比 v7 的 0.0100 稍差，因为预测更激进）
- **原始空间 RMSE**:
  - DeepLSTM_24h = **360.93**（比 Zero=116 差 3 倍）
  - DeepLSTM_48h = 377.94
  - DeepLSTM_full = 679.12
  - **Pred 值域: [0, 39811]** vs 真值 [0, 11903] — **峰值超出 3.3 倍，狂暴过预测**
- **GNN 叠加**:
  - DeepLSTM_GNN_24h = 285.06 (**+21% 比 LSTM 好**)
  - 81/83 县 GNN 帮忙 — 空间信息能削减过预测噪声
- **Detroit 24h RMSE**: 2326；Wayne 24h: **5397**（比 v7 更差）
- **诊断**: threshold=0.005 (0.5%) 太低，触发加权的样本远超 30%（日常小停电也触发）；weight=20 在原始空间被 tracked × 660K 放大后爆炸。**从"躺平"矫枉过正到"狂暴"**。

### 3.5 DeepLSTM v9: rate + weighted_mse, w=3, threshold=0.01（中庸也不行）

- **改动**: `EXTREME_WEIGHT = 3.0`, `EXTREME_THRESHOLD = 0.01`
- **训练日志**:
  - TrLoss epoch 1 = 0.0244 → 最终 0.0002
  - ValRMSE (rate) = **0.0099** ← 和 v7 的 0.0100 几乎一样（撞特征地板）
- **原始空间 RMSE**（第一次运行）:
  - DeepLSTM_24h = **129.71** (比 HistAvg 差 16 点)
  - DeepLSTM_48h = 122.80
  - DeepLSTM_full = 173.37
  - **Pred 值域（GNN）: [0, 6518]** — 向量值比 v7 合理但整体还是 miss time
- **GNN 叠加**（这次反而**伤害**）:
  - DeepLSTM_GNN_24h = **160.35** (-24% 比 LSTM 差)
  - 40/83 县 LSTM 更好，36/83 县 GNN 更好（基本随机）
- **诊断**: v7/v8/v9 的 rate-space ValRMSE 都落在 **0.0099 ± 0.002** 附近 — **这是特征能解释的信号上限**。换 loss 只能在 113-360 之间滑动，不能突破。模型因为没有"何时 storm"的时序信号，任何让它敢预测大值的 loss 都会把大值放错时间 → 原始空间更差。

### 3.6 HistGBM Baseline（诊断实验，成功）

- **Config**: `HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=8, min_samples_leaf=20, l2_regularization=1.0)`, log1p target, 同一套 Tier B 特征，同一套 interleaved split
- **Fit**: 4 秒, n_iter=150（early stopping）
- **Pred 值域**: [0, 4541] vs 真值 [0, 11903] — 峰值约真值 38%
- **原始空间 RMSE**:
  - HistGBM_24h = **92.17** ← 打败 HistAvg 21 点
  - HistGBM_48h = **80.81** ← 打败 HistAvg 21 点
  - HistGBM_full = 116.87
- **Per-county 灾区 (24h)**: Detroit 1154, Wayne 1022, Washtenaw 795, Gladwin 682 — **Detroit RMSE 比 LSTM 的 1807 降了 36%**
- **诊断**: 特征里**有**足够信号，HistGBM 能榨出来，LSTM 榨不出来。**架构问题 > 特征问题**。

---

## 4. 当前排行榜（原始空间，interleaved val）

### 24h

| 模型 | RMSE | vs HistAvg |
|---|---|---|
| **HistGBM** | **92.17** | **-18.8%** ✨ |
| HistAvg | 113.42 | baseline |
| Zero | 116.10 | +2.4% |
| DeepLSTM (v9) | 129.71 | +14.4% (worse) |
| DeepLSTM_GNN (v9) | 160.35 | +41% (much worse) |
| Persistence | 153.40 | +35% (worse) |

### 48h

| 模型 | RMSE | vs HistAvg |
|---|---|---|
| **HistGBM** | **80.81** | **-20.8%** ✨ |
| Zero | 100.46 | -1.5% |
| HistAvg | 102.00 | baseline |
| DeepLSTM (v9) | 122.80 | +20.4% (worse) |
| DeepLSTM_GNN (v9) | 147.13 | +44% (much worse) |

### Full (所有 val 时段)

| 模型 | RMSE |
|---|---|
| **HistGBM** | **116.87** |
| HistAvg | 148.36 |
| Zero | 148.94 |
| DeepLSTM | 173.37 ~ 284.11（随 seed 波动） |
| Persistence | 183.78 |

---

## 5. 关键发现与教训

### 发现 1: Split 策略比模型选择更重要
老的 chronological split 把风暴全扔给 val，让所有模型**看起来**过拟合；改成 interleaved split 后，v7 LSTM 的 "overfitting" 现象消失 (val RMSE 1.38 → 1.02)，才发现真问题是分布不匹配。

### 发现 2: rate 空间 + 标准 loss 会让模型躺平
70% 样本是 0 + rate 值域 [0, 1] + huber δ=0.01 → 模型的全局最优就是"一律预测 0"，TrLoss 立刻归 0，完全不学。

### 发现 3: weighted_mse 的甜区窄且难调
- w=1 (huber) → 躺平
- w=3 → val 打平 Zero 基线但原始空间仍比 HistAvg 差
- w=20 → 过度矫正，狂预测大值
- **问题本质**: rate 目标把 Detroit 从"绝对值大 10 倍"变成"和别人一样的 0-1 区间"，但 loss 无法告诉模型"**何时**该预测大值"。

### 发现 4: LSTM 的 ValRMSE 地板 ≈ 0.0099
v7/v8/v9 三个不同 loss 配置下，rate-space ValRMSE 都收敛到 0.0099 ± 0.002。**这是 Tier B 特征 + LSTM 架构能榨出的信号上限**。换 loss 只影响"这个地板映射到原始空间后的表现"，不影响地板本身。

### 发现 5: Detroit 效应支配县均 RMSE
Detroit (26125) 单县 RMSE 占总 RMSE 的 15-20%。所有模型最差的县都是 Detroit + Wayne + Washtenaw + Gladwin（前 4 城市）。后 79 个小县 RMSE 平均只有 40-50，已经很好。**大城市特殊处理是刚需**。

### 发现 6: HistGBM 50 点优势说明架构问题
相同特征（Tier B），相同 split，HistGBM=92 vs LSTM=143。50 点差距不是特征能解释的 — 是 LSTM 架构把信号损失掉了。
- 猜测原因: 48h 输入序列被 RNN 整合成 1 个 hidden state，最强的"最近 1-6h lag"信号被稀释
- 猜测原因: RNN 对突变点响应慢，擅长平滑信号，不擅长 storm 的陡升

### 教训 1: 不要乱改 loss
v6 → v7 → v8 → v9 这条路花了 3 天，结论是 "loss 调不动 LSTM"。应该早点切到 HistGBM 诊断，本来 30 分钟能得到的信息。

### 教训 2: 先跑 tabular baseline，再上深度模型
HistGBM 是任何 tabular 回归任务的**必备** baseline。不跑这个直接上 LSTM 是盲调。

---

## 6. 产物清单

### 保存的 Checkpoint / 预测
- `results/DeepLSTM_best.pt` — v9 LSTM 权重 (241K params)
- `results/DeepLSTM_val_pred.npy` — v9 LSTM val 预测 (T_val × 83)
- `results/DeepLSTM_GNN_best.pt` — v9 GNN 权重 (315K params)
- `results/DeepLSTM_GNN_val_pred.npy` — v9 GNN val 预测
- ⚠ HistGBM 目前**没保存**（每次 4 秒重训），下次改 §7.5 时加上 `joblib.dump`

### Git 提交历史 (model_deep_lstm.ipynb)
```
7b548eb Hist GBM baseline
4b6262d changed weight           # v9
511f012 changed loss function     # v8
49a5303 bug fixed
dc1ce4d update                   # v7
5cc1cd2 fix weighted MSE and add true baseline
01a3f2c indent
8f65c77 maybe it will work
be3ac1e changed split way        # interleaved split
6dbae32 compatibility problem solved
```

---

## 7. 下一步计划

### 今天 (周一 4/20)
- [x] 写这份 PROGRESS.md
- [ ] §7.5 HistGBM cell 升级: Action 1 超参调优 + Action 3A Detroit 样本加权
- [ ] Colab 跑一次，目标 HistGBM_24h < 85

### 周二 (4/21): 特征工程 v3
- [ ] 在 `build_features` 里加 3 组新特征:
  1. 天气变化率: `w_{feat}_diff_3h`, `w_{feat}_diff_6h`
  2. 天气极值窗: `w_{feat}_max_6h`, `w_{feat}_std_6h`
  3. 空间滞后: 上风向县过去 6h 停电平均
- [ ] 可选: 不对天气做 PCA，保留 top-10 原始维度
- [ ] 重跑 HistGBM，目标 <75

### 周三 (4/22): Test Set 推理 + Phase 2 开工
- [ ] 生成 `test_24h` / `test_48h` 预测 CSV (用最好的 HistGBM)
- [ ] 开始 Part II 发电机分配 (基于 HistGBM 预测)

### 周四 (4/23): Part II 完成
- [ ] 发电机分配算法（贪心 / 整数规划）
- [ ] Part II 可视化 + 评估

### 周五 (4/24): Report 草稿
- [ ] 写 §1 问题描述 + §2 方法
- [ ] 写 §3 实验 (参考本 PROGRESS.md)

### 周六 (4/25): Report 终稿 + Buffer
- [ ] 完善 §4 结论 + §5 讨论 + 限制
- [ ] 交付物打包

### 周日 (4/26): 提交

---

## 8. 待决定的问题

1. **是否用 LSTM 做 Ensemble 一员？**
   - 目前 LSTM=143, HistGBM=92。LSTM 对 HistGBM 的相对 hurt 不确定
   - 如果 LSTM 在某些县（比如小县）更好，可以 per-county stacking
   - **决策时点**: 周三，HistGBM 优化定稿后

2. **是否做 Two-Stage（先分类再回归）？**
   - 思路: 用 storm_indicator 特征训分类器判"未来 24h 是否有 storm"，只有有 storm 才走回归
   - 风险: 增加复杂度，deadline 前可能来不及
   - **决策时点**: 周二，如果 HistGBM 特征工程后还卡在 75-80

3. **Report 要不要保留 LSTM 内容？**
   - 当前倾向: 保留。作为"我们试了深度模型，发现架构瓶颈"的 narrative。现代 ML 课程看重**诚实汇报失败**
   - 但要简练，不能占超过 2 页

---

## 附录: 重要超参对照

| 参数 | v7 | v8 | v9 |
|---|---|---|---|
| TARGET_TYPE | rate | rate | rate |
| LOSS_FN | huber | weighted_mse | weighted_mse |
| HUBER_DELTA | 0.01 | - | - |
| EXTREME_WEIGHT | - | 20 | 3 |
| EXTREME_THRESHOLD | - | 0.005 | 0.01 |
| 其他 | 全部相同 | | |

公共: SEQ_LEN=48, HORIZON=48, HIDDEN_DIM=128, NUM_LAYERS=2, DROPOUT=0.4, LR=5e-4, WD=1e-3, BATCH=128
