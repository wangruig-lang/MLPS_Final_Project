# MLPS Final Project - Power Outage Forecasting

CMU 95-828 Machine Learning for Problem Solving - Spring 2026

## 项目概述

本项目分为两个部分：
1. **Part I: 短期停电预测** — 用历史停电数据 + 天气数据预测 Michigan 83 个县未来 24h/48h 的每小时停电数（评分占 30%）
2. **Part II: 备用发电机摆放** — 基于 Part I 的预测结果，决定 5 台备用发电机如何分配到各县以最大化停电缓解（评分占 30%，含 report 论述）

## 项目结构

```
├── demo.ipynb                    # 主 notebook（Part I + Part II 完整 pipeline）
├── requirements.txt              # Python 依赖列表
├── .env.example                  # 环境变量模板（复制为 .env 后填入自己的信息）
├── .env                          # 你自己的环境变量（不会被 git 追踪）
├── .gitignore
├── Final project description.pdf # 项目要求文档
├── Final Project Intro.pdf       # 项目介绍 slides
├── submission_template_24h.csv   # 24h 预测提交模板
├── submission_template_48h.csv   # 48h 预测提交模板
├── data/
│   ├── train.nc                  # 训练数据（需手动下载，~152MB）
│   ├── test_24h_demo.nc          # 24h 测试 demo 数据
│   └── test_48h_demo.nc          # 48h 测试 demo 数据
├── results/                      # 模型输出
│   ├── sarimax_pred_24h.csv      # SARIMAX 24h 预测
│   ├── sarimax_pred_48h.csv      # SARIMAX 48h 预测
│   ├── seq2seq_pred_24h.csv      # Seq2Seq 24h 预测
│   ├── seq2seq_pred_48h.csv      # Seq2Seq 48h 预测
│   ├── recommended_counties.txt  # Part II: 发电机分配决策文件（5 个 FIPS 码）
│   ├── weight_search.png         # Part II: weight search 可视化
│   ├── generator_allocation.png  # Part II: 分配结果可视化
│   └── generator_timeline.png    # Part II: 48h 时间线可视化
└── wandb/                        # W&B 本地日志（不会被 git 追踪）
```

## Notebook 结构说明

`demo.ipynb` 分为以下 sections：

| Section | 内容 | 说明 |
|---------|------|------|
| §1 配置 | 超参数 + wandb 初始化 | 所有可调参数集中在此 |
| §2 数据加载 | 加载 NetCDF 数据 | 包括 train/test 数据 |
| §3 EDA | 探索性数据分析 | 可视化停电分布和天气关联 |
| §4 数据准备 | 时间序列拆分 + 标准化 | 80/20 temporal split |
| §5 模型定义 | SARIMAX + Seq2Seq 定义 | 包括训练函数和评估函数 |
| §6 验证集评估 | 训练 + 验证 + 可视化 | **训练中实时监控 Val RMSE** |
| §7 测试集预测 | 最终模型训练 + 提交文件生成 | 输出 CSV 预测文件 |
| **§8 Part II** | **发电机摆放优化** | **weight search + 贪心分配 + 评估** |

### §8 Part II 详解：发电机摆放

这部分是我们的 **decision-aware ensemble** 方法：

1. **Weight Search**（Cell 59-60）
   - 24h 模型的前 24 小时预测通常更准（horizon 更短）
   - 48h 模型覆盖完整 48 小时但前 24h 可能不如 24h 模型
   - 混合公式：`hours 1-24: w × pred_24h + (1-w) × pred_48h[:24]`，`hours 25-48: pred_48h[24:48]`
   - 在验证集上搜索最优 `w`，**直接优化发电机缓解量**（不是 RMSE）
   - 输出：最优权重 `OPTIMAL_W` 和 search 曲线可视化

2. **贪心分配**（Cell 61）
   - 用最优 `w` 混合 test 预测
   - 逐台分配发电机到边际缓解量最大的县
   - 对比三种策略：Blended / 48h-only / Oracle（完美预测）

3. **评估与可视化**（Cell 62-63）
   - 条形图：top 15 县停电量 + 发电机标记
   - 三方对比柱状图
   - 48h 时间线：每个被分配县的真实/预测停电 + 容量线

4. **输出** → `results/recommended_counties.txt`（提交文件）

## 环境配置（首次使用必读）

### 第一步：克隆仓库

```bash
git clone <仓库地址>
cd MLPS_Final_Project
```

### 第二步：下载训练数据

`train.nc` 文件太大无法放在 GitHub 上，需要手动下载：

1. 从 Canvas 或共享的 Google Drive 下载 `train.nc`
2. 将文件放入 `data/` 目录下，即 `data/train.nc`

### 第三步：配置 Weights & Biases (wandb)

我们使用 wandb 来统一记录实验日志，方便团队成员之间对比不同 run 的结果。

1. **注册/登录 wandb 账号**：前往 https://wandb.ai 注册或登录
2. **创建或加入一个 Team**：wandb 要求使用 team entity 来记录 run。在 wandb 上创建一个 team（或让队长创建后邀请你加入），记住 team 名称
3. **获取 API Key**：登录后在 https://wandb.ai/authorize 页面复制你的 API Key
4. **创建 `.env` 文件**：

   ```bash
   cp .env.example .env
   ```

5. **编辑 `.env` 文件**，填入你自己的信息：

   ```
   WANDB_USERNAME=你的wandb用户名
   WANDB_API_KEY=你的wandb_api_key
   WANDB_ENTITY=你的team名称
   ```

> **注意**：`.env` 文件已被 `.gitignore` 忽略，不会被提交到 git，所以你的 API Key 是安全的。每个团队成员都需要各自创建自己的 `.env` 文件。所有团队成员的 `WANDB_ENTITY` 应填相同的 team 名称，这样所有 run 会汇总到同一个 project 下。

### 第四步：安装依赖并运行 notebook

打开 `demo.ipynb`，按顺序运行 cell 即可。notebook 的第一个代码 cell 会自动完成以下操作：

- **Conda 用户**：自动通过 conda 安装 `netCDF4` 和 `PyTorch`，再通过 pip 安装其余依赖
- **非 Conda 用户（如 Colab）**：通过 pip 安装所有依赖，PyTorch 从官方源安装

你不需要手动运行 `pip install`，notebook 会处理一切。

## Wandb 使用说明

### 自动记录的内容

| 类别 | 记录的指标 |
|------|-----------|
| **超参数** | random_seed, validation_split, sarimax_order, seq_len, batch_size, epochs, learning_rate, hidden_dim, num_layers, device |
| **训练过程** | 每个 epoch 的 train_loss、**train_rmse、val_rmse**、best_val_rmse、训练耗时 |
| **验证集评估** | SARIMAX / Seq2Seq / Zero Baseline 在 24h 和 48h 上的 RMSE |
| **测试集评估** | SARIMAX / Seq2Seq / Zero Baseline 在 24h 和 48h 上的 RMSE |
| **Weight Search** | 最优 w、search 曲线、每个 w 值的缓解量和 RMSE |
| **发电机分配** | blended/48h-only/oracle 三种策略的缓解量、mitigation rate、decision efficiency、分配表格、可视化图片 |

### Run 命名规则

每次运行会自动生成 run name，格式为：`{用户名}_{月日_时分秒}`

例如：`wangruig_0329_143025` 表示用户 wangruig 在 03 月 29 日 14:30:25 发起的 run。

### 查看实验结果

所有团队成员的 run 都会记录在同一个 wandb 项目 `MLPS-Power-Outage` 下，前往 wandb dashboard 即可查看和对比所有人的实验结果。

## 提交清单

| # | 文件 | 提交方式 |
|---|------|---------|
| 1 | Project Report (PDF, ≤8页) | Canvas |
| 2 | Jupyter Notebook (demo.ipynb) | Canvas |
| 3 | `*_pred_24h.csv` (1,992 行) | Google Form |
| 4 | `*_pred_48h.csv` (3,984 行) | Google Form |
| 5 | `recommended_counties.txt` (5 个 FIPS 码) | Google Form |

## 常见问题

### Q: `netCDF4` 安装失败

如果你使用的是 conda 环境，notebook 会自动通过 conda 安装。如果仍然失败，手动运行：

```bash
conda install -c conda-forge netcdf4
```

### Q: `torch` 安装失败 / 找不到版本

PyTorch 在某些平台和 Python 版本下无法通过 `pip install torch` 直接安装。notebook 已经处理了这个问题：
- Conda 用户会通过 conda 安装
- 其他用户会从 PyTorch 官方源安装

如果仍然失败，手动运行：

```bash
# conda 用户
conda install pytorch -c pytorch

# pip 用户 (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q: `FileNotFoundError: train.nc`

你需要手动下载 `train.nc` 并放入 `data/` 目录。详见上面的「第二步：下载训练数据」。

### Q: wandb 报错 `entity not found` 或 `403`

- 确保 `.env` 中的 `WANDB_ENTITY` 填的是 **team 名称**（不是个人用户名）
- 确保你已加入该 team
- 修改 `.env` 后需要 **重启 Jupyter Kernel** 才能生效

### Q: wandb 提示 credentials 找不到

确保你已经：
1. 复制了 `.env.example` 为 `.env`（`cp .env.example .env`）
2. 在 `.env` 中填入了正确的 `WANDB_USERNAME` 和 `WANDB_API_KEY`

如果不配置 wandb，notebook 仍然可以正常运行，只是不会记录实验日志。

## Deadline

April 26th, 2026 at 11:59 PM
