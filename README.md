# 第六届计图人工智能挑战赛：TempLR

[![Python](https://img.shields.io/badge/Python-3.7-green?style=flat-square)](https://www.python.org/)
![Jittor](https://img.shields.io/badge/Jittor-1.3.9.13-blue?style=flat-square)
![Task](https://img.shields.io/badge/Task-Temporal_Recommendation-orange?style=flat-square)

> 本项目聚焦时序图中的未来链接预测任务。给定用户、物品、作者、网页等实体在历史上的交互序列，模型需要预测未来最有可能发生的新连接。

## 目录

- [第六届计图人工智能挑战赛：TempLR](#第六届计图人工智能挑战赛templr)
  - [目录](#目录)
  - [项目结构](#项目结构)
  - [环境配置](#环境配置)
  - [系统配置](#系统配置)
  - [模型结构](#模型结构)
  - [训练配置](#训练配置)
  - [输出文件](#输出文件)
  - [训练](#训练)
  - [测试](#测试)
  - [实验结果](#实验结果)

## 项目结构

```text
Jittor/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml          # 默认运行参数，改这里即可调整默认值
├── src/
│   └── templr/               # 核心函数与训练、验证、提交逻辑
│       ├── core.py           # 采样、时间特征、MRR、日志、格式检查
│       ├── training.py       # 单数据集训练与推理
│       ├── submission.py     # 多数据集、多 seed 集成与打包
│       └── config.py         # 默认配置读取
├── scripts/
│   └── run.py                # 统一命令行入口
├── data_A/
│   ├── dataset1/
│   │   ├── train.csv
│   │   └── test.csv
│   └── dataset2/
│       ├── train.csv
│       └── test.csv
├── models/                   # 训练得到的模型权重
└── outputs/                  # 预测结果、日志、提交压缩包
```

## 环境配置

本项目使用 Python 3.7，主要依赖包括 numpy、jittor、jittor-geometric 和 PyYAML。

```bash
conda create -n Jittor python=3.7 -y
pip install -r requirements.txt
```

## 系统配置

默认使用 GPU 运行，配置位于 `configs/default.yaml`：

```yaml
common:
  use_cuda: 1
```

如需临时切换到 CPU，可把 `use_cuda` 改为 `0`，或在命令行传入：

```bash
python scripts/run.py train --dataset dataset1 --use_cuda 0
```

数据默认放在 `data_A/dataset1` 和 `data_A/dataset2` 下，每个数据集包含 `train.csv` 与 `test.csv`。

## 模型结构

工程支持两种主模型，并加入时间感知启发式特征进行候选排序融合：

- `baseline`：JittorGeometric 提供的基线模型。
- `mynet`：CTG-Ranker 实验网络，使用因果时间邻居编码、候选 destination 历史编码、pairwise history features 和 MLP re-ranker。

核心模块：

- `src/templr/training.py`：训练、验证、推理主流程。
- `src/templr/runtime.py`：训练循环、验证和批量推理工具。
- `src/templr/models/factory.py`：根据参数构建 `baseline` 或 `mynet`。
- `src/templr/models/baseline.py`：baseline 网络定义。
- `src/templr/models/mynet.py`：mynet 网络定义。
- `src/templr/baseline.py`：baseline/mynet 共用的训练、验证和比赛推理流程。
- `src/templr/core.py`：候选采样、时间特征、本地 MRR、日志与格式检查。
- `src/templr/submission.py`：多 seed 训练、预测集成和提交包生成。

## 训练配置

默认参数集中写在 `configs/default.yaml`。常用字段如下：

- `paths.model_dir`：模型权重输出目录，默认 `./models`。
- `paths.output_dir`：单独运行训练时的预测结果输出目录，默认 `./outputs`。
- `paths.submission_dir`：提交文件目录，默认 `./outputs/submission`。
- `run.default_command`：不带子命令运行 `scripts/run.py` 时默认执行的命令，当前为 `submit`。
- `run.model`：默认运行使用的主模型，当前为 `mynet`，可选 `baseline`。
- `mynet.*`：`mynet` 的模型参数，网络定义位于 `src/templr/models/mynet.py`。
- `baseline.*`：`baseline` 的模型参数，网络定义位于 `src/templr/models/baseline.py`。
- `mynet.hidden_size` / `baseline.hidden_size`：模型隐藏层维度。
- `mynet.shortcut_scale` / `baseline.shortcut_scale`：时间与重复交互捷径特征的缩放系数。
- `mynet.time_frequencies` / `baseline.time_frequencies`：Fourier 时间编码频率数量。
- `mynet.max_co_items` / `baseline.max_co_items`：co-occurrence 特征使用的源节点近期历史数量。
- `train.epochs`：最大训练轮数。
- `train.lr`：学习率。
- `train.batch_size`：批大小。
- `train.early_stop`：早停耐心轮数。
- `train.train_candidates`：训练时每条样本的候选数量。
- `train.val_candidates`：本地验证 MRR 的候选数量。
- `submission.seed`：一键提交时使用的随机种子。

命令行参数会覆盖 `configs/default.yaml` 中的默认值。

## 输出文件

提交命令默认输出如下：

```text
models/
└── dataset1_MYNET_best.pkl

outputs/
├── logs/
│   ├── train/
│   ├── check_data/
│   └── check_submission/
└── submission/
    ├── dataset1.csv
    ├── dataset2.csv
    └── result.zip
```

## 训练

单数据集训练并生成预测：

```bash
python scripts/run.py train --dataset dataset1
python scripts/run.py train --dataset dataset2
```

使用自定义配置文件：

```bash
python scripts/run.py train --config configs/default.yaml --dataset dataset1
```

一键训练两个数据集并打包：

```bash
python scripts/run.py submit
```

## 测试

检查数据格式：

```bash
python scripts/run.py check-data
```

计算本地启发式 MRR：

```bash
python scripts/run.py local-mrr --dataset dataset1
```

检查提交文件格式：

```bash
python scripts/run.py check-submission
```

## 实验结果

当前工程会在训练日志中记录每轮训练 loss、本地验证 MRR、融合权重和最终预测范围。正式实验结果可根据 `outputs/logs/train/` 中的日志补充到此处。
