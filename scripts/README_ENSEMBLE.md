# 用 Ensemble (best model) 跑出三个交付物 — 使用指南

## 你将得到

执行完后，`results/` 里会有：

| 文件 | 内容 |
|---|---|
| `results/ensemble_pred_24h.csv` | 24h 预测 (1992 行)，列 `timestamp,location,pred`，时间戳格式 `6/30/23 1:00` |
| `results/ensemble_pred_48h.csv` | 48h 预测 (3984 行) |
| `results/recommended_counties.txt` | `[fips, fips, fips, fips, fips]` |
| `results/direct_ensemble_models.pkl` | 训练好的 144 个模型 (XGB×48 + HistGBM×48 + ET×48) |
| `results/greedy_gain_curve_ensemble.csv` | 每台发电机的边际缓解曲线（写报告用） |

## 模型说明

报告 §5.4 的 ensemble：`ŷ = 0.5·XGB(log1p) + 0.2·HistGBM_scaled(log1p) + 0.3·ET(sqrt)`

**改动是把它扩成 Direct Multi-Horizon**：对每个 horizon `h ∈ {1..48}` 各训一组 `(XGB, HistGBM_scaled, ET)`，三模型结果按 `0.5/0.2/0.3` 加权。推理时用训练集最后一小时的 83 个县特征向量做单次前向，无 AR、无误差累积。

---

## Colab 运行步骤

> **前置**：`train.nc`, `test_24h_demo.nc`, `test_48h_demo.nc` 都已在
> `Google Drive / MyDrive / MLPS_Final_Project / data/` 下。

1. 在 Colab 打开 `Notebooks/model_deep_lstm_Colab.ipynb`
2. **Runtime → Change runtime type → T4 GPU**（XGB/ET 走 CPU 也行，但保留 GPU 不亏）
3. 顶部 **§0** 配置保持 `SKIP_LSTM_TRAIN=True`, `SKIP_PART2_GNN=True`（不跑 LSTM/GNN，只走 tabular pipeline）
4. **从头依次跑** §1.1 → §1.2 → ... → §7.5
   - 关键依赖：`§1.5` 产生 `feature_cols` `scaler`；`§7.5` 跑完后 `train_sorted` `val_sorted` `X_tr_tab` `X_va_tab` `y_tr_tab` 都在内存里
5. 跳到底部新增的三块（找标题 `## Ensemble (best model from report) — §7.11 / §8.1 / §9.1`）
6. **依次执行**：
   - `§7.11 Direct Multi-Horizon Ensemble`：训练 ~50–70 分钟（48 horizon × 3 模型 = 144 fits）
   - `§8.1 Test inference with ensemble`：~1 分钟，写 `ensemble_pred_24h.csv` / `ensemble_pred_48h.csv`
   - `§9.1 Greedy generator allocation`：~10 秒，写 `recommended_counties.txt`
7. 把 `results/` 里那 3 个文件下载到本地

> **如果 Colab 断了**：模型已存到 `results/direct_ensemble_models.pkl`，`§8.1` 会自动从 pkl 加载，不用重训。

> **如果想加速**：把 `§7.11` 里的 `n_estimators=600`（XGB）和 `n_estimators=200`（ET）调小一些。

---

## 本地验证

下载 `results/` 后，在项目根目录跑：

```bash
python scripts/verify_submission.py
```

会做以下检查（每条会打印 `OK` / `FAIL`）：

1. 两个 CSV 的列名是 `timestamp,location,pred`
2. 行数分别为 1992 / 3984
3. `pred` 没 NaN、全 ≥ 0、覆盖 83 个县和 24/48 个时刻
4. **timestamp 顺序与 `submission_template_24h.csv` / `submission_template_48h.csv` 完全一致**
5. **location 顺序与模板一致**
6. `recommended_counties.txt` 是 5 个整数的 list
7. **用 48h CSV 重跑贪心，结果必须和 .txt 里的县匹配**（这步是“数据决策一致性”证明）

最后给一行汇总：`VERIFY: ALL CHECKS PASSED` 或 `VERIFY: N FAILURE(S)` + 失败列表。

可选参数：

```bash
python scripts/verify_submission.py --pred-dir results --templates .
python scripts/verify_submission.py --p24 my24.csv --p48 my48.csv --counties picks.txt
```

---

## 常见问题

**Q: §8.1 报错 `reference rows X != 83`？**
A: §1.5 / §7.5 做 `dropna(subset=feature_cols)` 时把训练集最后一小时某些县丢了。把 §0 的 `OUTAGE_LAGS` 里最大值（24）调小，或回到 §1.5 用 `df_feat`（未 drop 的版本）。新 §8.1 会自动找 `df_feat → df → train+val concat`。

**Q: 验证脚本 `FAIL: greedy re-derived from 48h CSV matches declared counties`？**
A: 你两个文件不是同一次推理产出的（很可能 .txt 是旧版 §7.10 HistGBM-only 的结果，CSV 是新版 ensemble）。重跑 §9.1 即可。

**Q: 验证脚本 `FAIL: 24h timestamp order matches template`？**
A: 你提交的 CSV 时间戳格式不是 `6/30/23 1:00`，而是 `2023-06-30 01:00:00`。`§8.1` 已经写的是模板格式；如果是别的来源，用 pandas 重存：
```python
import pandas as pd
d = pd.read_csv("yours.csv")
d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%-m/%-d/%y %-H:%M")
d.to_csv("yours_fixed.csv", index=False)
```

**Q: 在哪里看到 ensemble 比单模型好？**
A: §7.11 末尾打印的表：

```
window     XGB    HGB_sc     ET    ENS
1-24h      ...    ...        ...   <-- 应该比任何一列都低
25-48h     ...    ...        ...
1-48h      ...    ...        ...
```

如果 ENS 没赢任一列，说明 0.5/0.2/0.3 权重对 multi-horizon 不是最优；可以在 §7.11 末尾自己扫一下 weights：
```python
for w in [(0.5,0.2,0.3),(0.4,0.3,0.3),(0.6,0.1,0.3),(0.5,0.3,0.2)]:
    rmses = [np.sqrt(np.mean((w[0]*p_xgb+w[1]*p_hgb+w[2]*p_et - yv_raw)**2)) for h in HORIZONS]
    ...
```
