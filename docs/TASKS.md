# MLPS Final Project — 分工表

**Deadline**: 2026-04-26 23:59 (剩 ~4 天)
**队伍**: 4 人 (P1/P2/P3/P4)
**当前进度**: Part I 主力模型训好 (Direct HistGBM), Part II 贪心决策已输出, 报告未写

---

## 现状快照

### 已完成
- **Part I 主力模型**: §7.10 Direct Multi-Horizon HistGBM (48 个模型, 1-48h 独立预测)
  - Chronological val: 24h=281 / 48h=277 (这个 val 含风暴, 困难)
  - Interleaved val (§7.6 D1): 24h=88.19 (温和分布参考值)
- **Part II 决策**: 贪心分配 → `[26125, 26125, 26163, 26163, 26139]`, 总缓解率 66.4%
- **提交文件**: `results/direct_pred_24h.csv`, `direct_pred_48h.csv`, `recommended_counties.txt`

### 待办 (按优先级)
1. **Part II 报告** (占总分 ~15%, 没写就零分) — P2 主导
2. **Part I 报告** (占总分 ~15%) — P2 主导
3. **Sensitivity 分析 + 可视化** (必加分项) — P4 主导
4. **DL 模型尝试** (可选, 高风险, 可能 +2-5 分) — P3 主导
5. **Ensemble + 最终提交验证** — P1 主导

---

## 分工

### P1 — 主力 pipeline & 最终集成 (队长)
**职责**: 守住主干, 保证最终能按时提交一套正确文件

**任务**:
- [ ] 验证当前 `direct_pred_*.csv` 能被 demo.ipynb 的 `evaluate_per_county` 正确读入 (15 min)
- [ ] 跑 SARIMAX + Direct ensemble (0.5/0.5 和其它权重扫描), 生成 `ensemble_pred_*.csv` (2h)
- [ ] 最终挑选提交版本, 确认 CSV 格式 (timestamp, location, pred) 无误
- [ ] 4/25 锁定最终提交, 4/26 上午留最后 buffer
- [ ] 在 README.md 里更新最终使用的模型和 RMSE

**产出**:
- `results/ensemble_pred_24h.csv`, `ensemble_pred_48h.csv`
- `results/final_pred_24h.csv`, `final_pred_48h.csv` (选定方案)

---

### P2 — 报告撰写 (Part I + Part II)
**职责**: 把代码做的事情翻译成招生官/教授能看懂的学术文档

**任务**:
- [ ] **Part I 报告** (~5 页): 数据探索, 特征工程 (v2 精简), 模型对比 (Persistence/HistAvg/SARIMAX/Seq2Seq/HistGBM), 最终方案 Direct Multi-Horizon HistGBM 的理由
  - 关键论点: 为什么用 sqrt 变换 (storm-tail 问题), 为什么 Direct 不用 AR (compounding), HistGBM vs Seq2Seq 实证对比
  - 引用 `docs/PROGRESS.md` 的所有实验记录
- [ ] **Part II 报告** (~3-5 页): 问题建模, submodular 目标, 贪心算法, 结果解读, Sensitivity
  - 问题定义: `max sum_t min(alloc_i * 1000, y[t,i])`, submodular → greedy 有 (1-1/e) 保证
  - Rationale: 为什么用 Direct 48h 预测驱动决策, 不用 24h only
  - Sensitivity: 预测 ±30% 误差下决策是否变化 (P4 提供数据)
  - Oracle gap: 真值决策 vs 我们决策, 效率 %
- [ ] 整合成一份 PDF (LaTeX 或 Word 都行)

**产出**:
- `docs/report_part1.pdf` (或 .tex / .docx)
- `docs/report_part2.pdf`
- 或合并成 `docs/final_report.pdf`

**交付时间线**:
- 4/23 晚: Part II 初稿
- 4/24 晚: Part I 初稿
- 4/25 晚: 全文终稿

---

### P3 — 深度学习模型探索 (可选加分项)
**职责**: 尝试用 DL 超越 HistGBM, **不成功也没关系, 关键是别挤占 P1/P2 时间**

**任务**:
- [ ] 路线 A (推荐): 激活 `model_deep_lstm_Colab.ipynb` §P2 里的 LSTM+GNN
  - 改 `SKIP_LSTM_TRAIN=False`, `SKIP_PART2_GNN=False`
  - 调整 loss: 加 storm-weighted MSE (样本权重 = `1 + y/200`)
  - 训练 30-50 epochs, 目标: val RMSE < 275 (打败 Direct)
- [ ] 路线 B (备选): 写 Transformer multi-horizon (如果 A 失败)
- [ ] **硬 deadline**: 4/24 晚之前决定 GO/NO-GO, 若没超越 Direct 就放弃, 不进最终提交

**产出** (如果 GO):
- `results/lstm_gnn_pred_24h.csv`, `lstm_gnn_pred_48h.csv`
- `results/dl_model_best.pt`
- 2 页实验日志给 P2 放进 report

**产出** (如果 NO-GO):
- 1 页"我们尝试了什么、为什么没用"给 P2 放进 report (仍能拿分)

---

### P4 — Part II 分析 & 可视化
**职责**: 把 P1 的决策数据包装成有说服力的分析, 直接供 P2 写报告

**任务**:
- [ ] **Sensitivity 分析** (必做):
  - 预测扰动 ±10%, ±30% 下, 贪心分配是否改变
  - blend weight w ∈ {0, 0.25, 0.5, 0.75, 1.0} 扫描, 决策稳定性
  - 产出: 一张 heatmap 或 table
- [ ] **Oracle 对比**:
  - 用真值 (demo test 有 out 字段但是 noise, 可以做 synthetic storm 场景) 算 oracle 分配
  - 计算我们决策的 regret (oracle 缓解 − 我们缓解) / oracle 缓解
- [ ] **可视化**:
  - 83 个县的 48h 预测热力图 (timestamp × county)
  - Top 10 county 的预测曲线 + 5 台发电机位置标注
  - 贪心 gain 曲线 (每加一台发电机, 增量缓解)
- [ ] **Alternative 决策对比**:
  - 按人口排前 5 (naive baseline) vs 我们的
  - 按历史均值排前 5 vs 我们的
  - 说明"数据驱动决策"比 naive 好多少

**产出**:
- `results/part2_sensitivity.png`, `part2_allocation_viz.png`, `part2_county_heatmap.png`
- 一份 markdown/notebook: `docs/part2_analysis.ipynb` (给 P2 参考)

---

## 文件整理

### 建议删除/归档
```
# 无用
.Rhistory                          删除
.DS_Store                          删除 (加 .gitignore)
output/                            空目录, 删除
wandb/                             本地日志, .gitignore 里已有
```

### 建议保留
```
model_deep_lstm_Colab.ipynb       ← P1/P3 主战场
demo.ipynb                        ← 提交格式参考
phase1_eda.ipynb                  ← Phase 1 探索 (报告素材)
comparison_model.ipynb            ← baseline 对比
model_template.ipynb              ← 模板, 可选参考

data/                             ← 不要动
results/                          ← P1 输出
docs/
  PROGRESS.md                     ← 实验日志 (P2 写 report 的原料)
  TASKS.md                        ← 本文件
  part2_analysis.ipynb            ← P4 产出
  report_part1.pdf                ← P2 最终产出
  report_part2.pdf                ← P2 最终产出
```

### 建议废弃但保留历史
```
model_deep_lstm.ipynb             ← Colab 版本的本地副本, 让 P3 改这个或 Colab 版
model_template.ipynb              ← 早期骨架, 不删不用
```

---

## 日程表

| 日期 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| 4/22 (周三) | ensemble 实验 | Part II 报告提纲 | 评估 DL 路线可行性 | sensitivity 跑通 |
| 4/23 (周四) | ensemble 调参 | Part II 初稿 | DL 训练中 | 可视化图出 |
| 4/24 (周五) | ensemble 终版 | Part I 初稿 | **DL GO/NO-GO 决策** | oracle 对比 |
| 4/25 (周六) | 最终 CSV 锁定 | 报告终稿 | DL 集成 (若 GO) | 分析终稿 |
| 4/26 (周日) | 提交 + buffer | 提交文档校对 | — | 支援 |

---

## 沟通约定

- 每天晚上 10 点同步一次进展 (微信/Slack)
- 紧急 bug 随时互相 @
- 报告素材放 `docs/`, 模型输出放 `results/`, 不要动 `data/`
- Notebook 改动前先 git pull, 避免冲突 (Colab 版本 `model_deep_lstm_Colab.ipynb` 主要是 P1 操作, 其他人尽量在本地副本实验)

---

## 风险点

1. **报告不写**: 最大风险, 直接丢 30-40 分。P2 必须保证最晚 4/25 晚出终稿
2. **DL 占用时间**: P3 严格 4/24 晚 deadline, 不行就放弃
3. **最后一天提交文件格式错**: P1 务必 4/24 前用 demo.ipynb 的 evaluator 验证一次
4. **git 冲突**: 四人同时改 notebook 容易冲突, 建议每人改固定区域
