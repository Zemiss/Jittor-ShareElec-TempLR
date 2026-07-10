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
├── tests/
│   └── test_temporal_upgrade.py # 时序特征、门控与指标回归测试
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

本项目使用 Python 3.7，主要依赖包括 numpy、pandas、scikit-learn、jittor、jittor-geometric 和 PyYAML。

```bash
conda create -n jittor_env python=3.7 -y
conda activate jittor_env
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

工程支持两种主模型。`baseline` 保持原始实现不动，`mynet` 用于在 baseline 基础上做轻量实验：

- `baseline`：JittorGeometric 提供的基线模型。
- `mynet`：与 baseline 相同的 CRAFT 主干副本，额外在训练/预测流程中支持更多负样本和 learned rerank 后处理。

当前最有效的 `mynet` 改动是 `use_rerank: true`。CRAFT 主干本身负责候选感知交叉注意，二阶段 reranker 再使用神经网络候选分数、链路历史、候选流行度、近期历史命中和时序共邻居特征，对 100 个候选进行轻量重排。

本次按 `docs/develop.md` 完成的升级包括：

- seen/new 双头门控：同一个线性排序器为重复候选和新候选学习独立特征权重，并联合校准分数。
- TAMI 风格可选特征：提供对数时间间隔、近期交互频率、链路间隔稳定性和时间衰减共邻居；消融未通过时可切回 `basic`。
- 场景化路由：一键提交根据重复边画像让 dataset1 使用双头、dataset2 使用单头。
- 安全回退：holdout 调参始终包含原始神经分数，reranker 在本地 MRR 不提升时自动退回原排序。
- 诊断与效率：日志增加 Hits@1/3/10、median rank、repeat/new、short/long-gap、head/tail 切片，并缓存共邻居统计。

核心模块：

- `src/templr/training.py`：训练、验证、推理主流程；包含 `mynet` learned rerank 逻辑。
- `src/templr/runtime.py`：训练循环、验证和批量推理工具。
- `src/templr/models/factory.py`：根据参数构建 `baseline` 或 `mynet`。
- `src/templr/models/baseline.py`：baseline 网络定义。
- `src/templr/models/mynet.py`：mynet 网络定义。
- `src/templr/baseline.py`：baseline/mynet 共用的训练、验证和比赛推理流程。
- `src/templr/core.py`：候选采样、时间特征、本地 MRR、日志与格式检查。
- `src/templr/submission.py`：多数据集训练、预测和提交包生成。

## 训练配置

默认参数集中写在 `configs/default.yaml`。常用字段如下：

- `paths.save_model_dir`：模型权重输出目录，默认 `./models`。
- `paths.use_model_dir`：测试时读取模型权重的目录，默认 `./models`。
- `paths.output_dir`：单独运行训练时的预测结果输出目录，默认 `./outputs`。
- `paths.submission_dir`：提交文件目录，默认 `./outputs/submission`。
- `run.default_command`：不带子命令运行 `scripts/run.py` 时默认执行的命令，当前为 `submit`。
- `run.model`：默认运行使用的主模型，当前为 `mynet`，可选 `baseline`。
- `mynet.*`：`mynet` 的模型参数，网络定义位于 `src/templr/models/mynet.py`。
- `baseline.*`：`baseline` 的模型参数，网络定义位于 `src/templr/models/baseline.py`。
- `mynet.hidden_size` / `baseline.hidden_size`：模型隐藏层维度。
- `mynet.max_co_items` / `baseline.max_co_items`：co-occurrence 特征使用的源节点近期历史数量。
- `mynet.num_negatives`：`mynet` BPR 训练时每个正样本使用的随机负样本数。
- `mynet.use_rerank`：是否启用 learned rerank 后处理。
- `mynet.rerank_dual_head`：重复边/新边双头门控；默认 `auto`，dataset1 开启、dataset2 关闭，也可设为 `true/false`。
- `mynet.rerank_feature_set`：`basic` 为消融保留的稳定特征，`enhanced` 额外启用对数时间与间隔特征。
- `mynet.rerank_regularization`：二阶段 LogisticRegression 的正则化强度 `C`。
- `train.train_candidates`：训练时每条样本的候选数量。
- `train.val_candidates`：本地验证 MRR 的候选数量。
- `train.tune_val_samples`：用于验证和 rerank 调参的验证样本数。
- `submission.seed`：一键提交时使用的随机种子。

`submission.py` 还内置了数据集级策略：dataset1 默认使用双头门控，dataset2 使用单头；原有 blend 策略保持不变。`submit --rerank_dual_head true/false` 可以显式覆盖场景路由。

命令行参数会覆盖 `configs/default.yaml` 中的默认值。

## 输出文件

提交命令默认输出如下：

```text
models/
└── dataset1_MYNET_best.pkl

outputs/
├── logs/
│   ├── dataset1_*.log
│   └── dataset2_*.log
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

只运行已有 checkpoint 的 rerank holdout 消融，不生成完整 test 预测：

```bash
python scripts/run.py test --dataset dataset1 --validation_only true
python scripts/run.py test --dataset dataset1 --validation_only true \
  --rerank_feature_set enhanced --rerank_dual_head false
```

运行回归测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

检查提交文件格式：

```bash
python scripts/run.py check-submission
```

## 实验结果

当前工程会在训练日志中记录每轮训练 loss、本地验证 MRR/AP/AUC、rerank holdout 提升和最终预测范围。已验证的简单改动如下：

- baseline：test.csv 得分约 `1.0111`。
- `mynet.use_rerank: true`：test.csv 得分约 `1.240`，是当前收益最大的改动。
- `mynet.num_negatives: 100`：test.csv 得分约 `1.0682`，单独有小幅提升。
- `objective: sampled_softmax` + `selection_metric: MRR`：test.csv 得分约 `0.5534`，效果较差，当前不推荐。

本次升级使用已有 v2 checkpoint、固定 seed `2411273` 和 5000 条 validation query 做同条件消融：

| 数据集 | rerank 配置 | holdout MRR | 结论 |
|---|---|---:|---|
| dataset1 | basic + 单头（v2） | 0.874711 | 对照 |
| dataset1 | basic + 双头 | **0.875866** | 保留并用于场景路由 |
| dataset1 | enhanced + 单头 | 0.874616 | 不设为默认 |
| dataset1 | enhanced + 双头 | 0.875204 | 不设为默认 |
| dataset2 | basic + 单头 | **0.323886** | 保留并用于场景路由 |
| dataset2 | basic + 双头 | 0.323168 | 不保留 |

这些是本地 holdout 指标，不等同于新的 test.csv 榜单成绩；新版本提交前仍需按比赛评测确认。
