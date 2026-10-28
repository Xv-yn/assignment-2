# COMP4328 / COMP5328 Assignment 2: learning with noisy labels

本项目在原仓库的数据审计基础上，实现两种抗标签噪声分类器、噪声转移矩阵估计、独立随机 80/20 划分、重复实验、结果汇总、预测和源码打包。只使用给定的三个数据集，不下载额外数据或预训练权重。

## 1. 环境与数据

推荐 Python 3.10 或更新版本。早期 CPU 流程验证环境：Python 3.10.19、PyTorch 2.5.1 CPU、NumPy 2.0.1、Pillow 11.1.0。正式实验已在 A100 80GB、Python 3.12.3、PyTorch 2.8.0+cu128 上完成，结果见第 9 节。

在本 README 所在目录运行所有命令：

```bash
python -m pip install -r requirements.txt
```

支持 CPU 和 CUDA。GPU 安装方式参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)；`--device cuda` 会在 CUDA 不可用时明确报错，`auto` 自动选择。无需 torchvision、scikit-learn 或 matplotlib。

正式提交解压后，将原始文件放入 `data/`：

```text
data/FashionMNIST0.3.npz
data/FashionMNIST0.6.npz
data/CIFAR.npz
```

本地现有数据位于仓库的上级目录 `../datasets/`，因此以下示例显式使用 `--data-dir ../datasets`。交作业时可省略此参数，默认读取仓库内的 `data/`。

每个归档必须包含 `Xtr, Str, Xts, Yts`。FashionMNIST 的 `Xtr` 为 `(18000,784)`，CIFAR 为 `(15000,32,32,3)`；每个测试集有 3000 个样本。保留已有严格数据验证与整数类别 ID，不重新推断类别语义。

## 2. 快速验证

```bash
python -m unittest discover -s algorithm/tests -v
python -m algorithm.run_experiments --config algorithm/configs/smoke.json --data-dir ../datasets
python -m algorithm.summarize --results-dir results/smoke
python -m algorithm.run_experiments --config algorithm/configs/training_smoke.json --data-dir ../datasets
python -m algorithm.summarize --results-dir results/training_smoke --allow-incomplete
```

`smoke.json` 仅检查数据和生成带噪训练样例图。`training_smoke.json` 使用每组 192 个训练样例、48 个验证样例、一轮训练、小网络和一个随机种子，运行三组数据上的两种分类器，并最终在完整测试集上检查评估路径。它只用于检查程序，**不能用于报告的正式精度、矩阵估计或鲁棒性结论**。调试结果始终标为 `debug=true`，汇总要求显式加 `--allow-incomplete`。

## 3. 正式实验（60 次分类器训练 + 30 次矩阵估计）

```bash
python -m algorithm.run_experiments --config algorithm/configs/final.json --plan
python -m algorithm.run_experiments --config algorithm/configs/final.json --data-dir ../datasets
python -m algorithm.summarize --results-dir results/final
```

默认配置：3 个数据集 x 2 个分类器 x 10 个不同划分种子；每次最多训练 30 轮，patience=8。训练样本数分别为 14400 和 12000，验证样本数分别为 3600 和 3000。所有划分均为随机不放回的 80/20 分割，不分层；不同种子的划分允许重叠。每个划分上的两个分类器共用分割、初始化种子和批次顺序，方便配对比较。估计器单独初始化，最多训练 20 轮。

网络是从头训练的 SmallCNN：默认通道 32/64/128，每块有两层 3x3 卷积、BatchNorm、ReLU 和最大池化；随后 3x3 自适应池化和带 dropout 的全连接分类头。池化通过相同平均区域的显式切片实现，以支持严格确定性 CUDA 反向传播；测试已对照原生自适应池化的输出和输入梯度。Adam、初始学习率 0.001、weight decay 0.0001、余弦学习率调度。固定像素除以 255 转为 float32 NCHW，不进行统计标准化或数据增强。网络结构和优化设置对两种分类器一致。

可单独运行某组任务：

```bash
python -m algorithm.run_experiments --config algorithm/configs/final.json --data-dir ../datasets --dataset CIFAR --method forward --seed 0 --device cpu
python -m algorithm.run_experiments --config algorithm/configs/final.json --data-dir ../datasets --resume
```

第一次也使用相同的 `--device` 覆盖值，才能在后续以相同配置续跑。`--resume` 校验配置、代码、数据、环境和产物哈希，只跳过校验成功的完整任务；未完成任务从头重训，不从中间 epoch 恢复。修改代码、配置、设备或依赖后应换用 `--results-dir`。不同子集任务可在同一目录顺序补齐，但不要同时写入同一个任务目录。

CPU 可运行全部流程；完整实验耗时取决于硬件，建议使用可用 CUDA 环境。正式结果必须等全部任务完成后再汇总，不应使用测试集结果修改超参数。

## 4. 算法与防泄漏约定

统一矩阵方向：`T[i,j] = P(S=j | Y=i)`，其中 Y 是干净类别，S 是带噪标签。若网络预测行向量 `p=P(Y|x)`，带噪后验为 `p @ T`。最终预测使用 `argmax(p)`，不在干净测试时乘 T。

### Forward correction

目标：`L_forward = -log((softmax(logits) @ T)[s])`。实际代码用 logsumexp 实现稳定的前向损失。FashionMNIST0.3 和 FashionMNIST0.6 分别使用说明中提供的已知矩阵；CIFAR 使用该次划分独立估计的矩阵。

来源：[Patrini et al., CVPR 2017, Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html)。

### GCE 与显式的重要性加权扩展

基础 GCE：`L_q(p,s) = (1-p_s^q)/q`，默认 q=0.7。q 趋于 0 时趋向 CE，q=1 时为一半的多类别 L1/MAE 损失。通过降低低置信度带噪标签的梯度影响，提高对标签噪声的抵抗能力。

为了在未知翻转率任务中也实际使用估计的 T，**默认 `gce` 方法是重要性加权 GCE 扩展**：

```text
w(x,s) = stop_gradient(p_s / max((p @ T)_s, 1e-8))
L = min(w, 10) * (1 - p_s^0.7) / 0.7
```

默认前 3 轮使用 Forward CE 预热；之后切换到加权 GCE。重要性比率不参与求导，并进行上界截断。只有比率由真实后验给出且不截断时，重要性加权的期望恒等式才精确；这里是模型的近似估计，**不宣称它继承原版 GCE 的所有理论保证**。该扩展必须在报告中明确写出，不能将结果标为未经修改的原版 GCE。

将配置中 `method_parameters.gce.reweight` 设为 false、`warmup_epochs` 设为 0，可运行原版 GCE 消融。原版训练损失不使用 T，但本项目仍使用 T 转换的带噪后验做验证选模。请为消融使用新的结果目录。

基础 GCE 来源：[Zhang & Sabuncu, NeurIPS 2018, Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels](https://arxiv.org/abs/1805.07836)。**本目录未包含课件，无法确认 GCE 是否满足“至少一种算法未在课堂讲授”的条件；提交前必须结合课程材料核实。**

### 转移矩阵估计

在当前 80% 训练分区上用普通带噪 CE 训练辅助网络，只通过 20% 带噪验证分区的 CE 选择其检查点。恢复最佳检查点后，按原始顺序预测该训练分区的带噪后验 `g(x)=P(S|x)`。

对每个类别 i，选择 `g_i(x)` 最大的一小部分样本（默认前 1%），对这些样本的完整后验向量取平均作为 T 的第 i 行。`top_fraction=0` 表示每类只选一个 argmax 锚点；`quantile<1` 可排除极端置信度样例，默认不排除。所有锚点索引均保存到原始 Xtr 行号，估计过程不读取 Xts/Yts。两种分类器使用同一划分的同一估计矩阵。

该方法依赖存在近似纯类别锚点、后验估计较好以及干净/带噪类别 ID 对齐等假设。未知非对角占优噪声下类别排列不能仅靠带噪数据保证识别。程序记录条件数和对角占优诊断，不通过测试标签排列矩阵，也不将非对角占优估计强制改成已知模板。FashionMNIST0.6 的高噪声使后验类别间差距很小，矩阵估计可能不稳定；应根据正式实验讨论局限。

对两个已知矩阵数据集也运行估计器，记录逐元素 MAE 与 Frobenius 误差，但分类器仍使用给定真实矩阵。CIFAR 的正式矩阵需要完整训练后读取结果，不能使用 debug 矩阵代替。

### 选模与评估

分类器每轮只计算带噪验证集的 Forward NLL `-log((p @ T)_s)`，以它选择最佳 epoch 并触发早停。原始 GCE 和 CE 的损失数值不可直接比较，验证指标对两个分类器统一。恢复最佳模型后才加载干净测试集并计算 top-1；训练、早停、超参数选择和矩阵估计不访问干净测试标签。数据审计可读取测试数组检查形状和数值，但不训练或选模。

## 5. 结果和报告取数

```text
results/final/<dataset>/seed_000/
  split.npz                    # 完整 80/20 原始行号
  estimator/
    best.pt, history.json
    transition.json            # 矩阵、锚点、诊断、已知矩阵误差
    run.json                   # 运行状态和哈希清单
  forward/ (以及 gce/)
    best.pt, history.json
    metrics.json               # 精度、混淆矩阵、最佳轮数、训练时间
    predictions.npz            # 最终干净测试后验和真实标签
    run.json
```

汇总命令验证每个产物的哈希、运行来源及预测与精度是否一致，输出：

- `summary.md/json/csv`：每数据集/方法的精度均值、样本标准差（ddof=1）。
- `runs.csv`：逐次结果，可用于报告表格及进一步分析。
- `summary.json` 中 `transition_estimates`：逐划分矩阵的均值/标准差，以及两个已知矩阵上的估计误差。
- `summary.json` 中 `paired_comparisons`：相同划分上 GCE 减 Forward 的精度差（百分点）的均值/标准差。
- `accuracy_comparison.png`、`<dataset>_transition.png` 和每次训练的 `learning_curve.png`。

CIFAR 每次训练使用的是当次估计矩阵；报告可以展示十次估计的均值和标准差，但该均值不是所有运行共用的输入矩阵。正式汇总必须包含配置的完整 >=10 次划分网格；调试或缺少任务只允许输出清楚标记的 provisional 汇总。一次运行时标准差为 null/N/A，不伪造零波动。

## 6. 独立预测

输入 `.npy` 为原始 uint8 批次，形状与相应数据集的原始图像一致：

```bash
python -m algorithm.predict --checkpoint results/final/CIFAR/seed_000/forward/best.pt --images images.npy --output predictions.npz
```

保存干净类别 ID `predictions` 及 `probabilities`。仅加载自己生成或可信的检查点；使用 PyTorch 的 `weights_only=True`。

## 7. 源码提交包

```bash
python -m algorithm.package_submission --output dist/GroupIDXX_StudentIDxxxxxxxx_StudentIDxxxxxxxx_StudentIDxxxxxxxx.zip
```

请将示例文件名替换为真实组号和所有成员学号。压缩包包含 `algorithm/`、README、requirements 和**空的** `data/`，不包含数据、结果、模型权重、Git 目录或缓存。报告 PDF 需单独按说明命名。报告附录可以引用本 README 的安装、正式训练和汇总命令。

## 8. 文件职责

| 模块 | 职责 |
| --- | --- |
| data.py / audit_data.py | 原始数据严格验证、确定性预处理和审计 |
| models.py / losses.py | 共用 CNN、Forward、GCE 和重要性加权扩展 |
| transition.py | 真实矩阵、锚点估计和矩阵约束检查 |
| training.py | 随机种子、分割、训练、验证与评估 |
| experiments.py / run_experiments.py | 实验调度、隔离测试访问、状态记录与续跑 |
| results.py / summarize.py | 完整性检查、均值/标准差和图表 |
| provenance.py | 原子 JSON 写入及内容哈希 |
| predict.py / package_submission.py | 独立预测和源码打包 |
| tests/ | 变换/标签、测试泄漏、数学方向、梯度、锚点、分割和检查点回归测试 |

## 9. 已完成的正式结果

2026 年 10 月 28 日已完成全部 **60 次分类器训练和 30 次矩阵估计**，正式训练与汇总耗时约 18 分 41 秒。服务器上 38 项测试通过。以下为十次随机划分的干净测试集 top-1 精度，格式为均值 ± 样本标准差（%，ddof=1）；GCE 指本 README 说明的重要性加权扩展。

| 数据集 | Forward | 重要性加权 GCE |
| --- | ---: | ---: |
| FashionMNIST0.3 | 99.063 ± 0.136 | 98.997 ± 0.119 |
| FashionMNIST0.6 | 93.087 ± 1.548 | 92.137 ± 2.240 |
| CIFAR | 70.513 ± 6.684 | 64.440 ± 6.714 |

[正式结果与矩阵估计](experiment_results/final/summary.md)、[逐次精度](experiment_results/final/runs.csv)及[运行与备份记录](docs/REMOTE_RUN_NOTES.md)已保存。`experiment_results/final/` 提供精简公开证据，包含完整来源与产物哈希记录；模型权重、预测数组、数据和完整运行日志保留在本地 `dist/COMP5328_A2_server_backup.tar.gz` 及其校验后的解包目录中，不纳入 GitHub 精简结果。下载备份的 556 个文件均已通过 SHA-256 校验。
