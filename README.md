# MLPS Final Project - Power Outage Forecasting

CMU 95-828 Machine Learning for Problem Solving - Spring 2026

## 项目结构

```
├── demo.ipynb                    # 主 notebook（数据加载、EDA、模型训练、评估、提交）
├── requirements.txt              # Python 依赖列表
├── .env.example                  # 环境变量模板（复制为 .env 后填入自己的信息）
├── .env                          # 你自己的环境变量（不会被 git 追踪）
├── .gitignore
├── submission_template_24h.csv   # 24h 预测提交模板
├── submission_template_48h.csv   # 48h 预测提交模板
├── data/
│   ├── train.nc                  # 训练数据（需手动下载，~152MB）
│   ├── test_24h_demo.nc          # 24h 测试 demo 数据
│   └── test_48h_demo.nc          # 48h 测试 demo 数据
├── results/                      # 模型预测输出
└── wandb/                        # W&B 本地日志（不会被 git 追踪）
```

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
2. **获取 API Key**：登录后在 https://wandb.ai/authorize 页面复制你的 API Key
3. **创建 `.env` 文件**：

   ```bash
   cp .env.example .env
   ```

4. **编辑 `.env` 文件**，填入你自己的信息：

   ```
   WANDB_USERNAME=你的wandb用户名
   WANDB_API_KEY=你的wandb_api_key
   ```

> **注意**：`.env` 文件已被 `.gitignore` 忽略，不会被提交到 git，所以你的 API Key 是安全的。每个团队成员都需要各自创建自己的 `.env` 文件。

### 第四步：安装依赖并运行 notebook

打开 `demo.ipynb`，按顺序运行 cell 即可。notebook 的第一个代码 cell 会自动完成以下操作：

- **Conda 用户**：自动通过 conda 安装 `netCDF4` 和 `PyTorch`，再通过 pip 安装其余依赖
- **非 Conda 用户（如 Colab）**：通过 pip 安装所有依赖，PyTorch 从官方源安装

你不需要手动运行 `pip install`，notebook 会处理一切。

## Wandb 使用说明

### 自动记录的内容

notebook 会自动将以下内容记录到 wandb：

| 类别 | 记录的指标 |
|------|-----------|
| **超参数** | random_seed, validation_split, sarimax_order, seq_len, batch_size, epochs, learning_rate, hidden_dim, num_layers, device |
| **训练过程** | Seq2Seq 每个 epoch 的 loss 和训练耗时 |
| **验证集评估** | SARIMAX / Seq2Seq / Zero Baseline 在 24h 和 48h 上的 RMSE |
| **测试集评估** | SARIMAX / Seq2Seq / Zero Baseline 在 24h 和 48h 上的 RMSE |

### Run 命名规则

每次运行会自动生成 run name，格式为：`{用户名}_{月日_时分秒}`

例如：`wangruig_0329_143025` 表示用户 wangruig 在 03 月 29 日 14:30:25 发起的 run。

这样团队成员可以在 wandb dashboard 上一目了然地看到是谁在什么时间运行的实验。

### 查看实验结果

所有团队成员的 run 都会记录在同一个 wandb 项目 `MLPS-Power-Outage` 下，前往 wandb dashboard 即可查看和对比所有人的实验结果。

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
conda install pytorch cpuonly -c pytorch

# pip 用户
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q: `FileNotFoundError: train.nc`

你需要手动下载 `train.nc` 并放入 `data/` 目录。详见上面的「第二步：下载训练数据」。

### Q: wandb 提示 credentials 找不到

确保你已经：
1. 复制了 `.env.example` 为 `.env`（`cp .env.example .env`）
2. 在 `.env` 中填入了正确的 `WANDB_USERNAME` 和 `WANDB_API_KEY`

如果不配置 wandb，notebook 仍然可以正常运行，只是不会记录实验日志。

## Deadline

April 26th, 2026 at 11:59 PM
