# Industrial & Scientific：Base → CPT → SFT → RL（Apple MPS）

本页记录当前 Industrial & Scientific 实验；它与早期 Office Products chrono5000 实验是不同的数据集。模型为 **Qwen/Qwen2.5-1.5B Base**，不是 Instruct 版。代码、配置说明和汇总指标入库；原始数据、外部 embedding、下载模型与检查点不入库。

## 数据和 SID

使用外部提供的 25,847 件商品、256 维 embedding 及配套历史行为。不能仅根据文件名推断该 embedding 的模型、融合方式或图片覆盖率。商品映射、SID 与用户历史须保持同一套行号/ASIN 对应关系；不使用 Office Products 的 5,000 商品子集替代。

训练/验证/测试交互样本分别为 **259,983 / 50,985 / 50,985**，沿用外部行为数据的划分，逐用户留出后续目标，不跨集合回填训练标签。输入文件不随 Git 分发；准备好的输入清单位于本机 `data/Amazon23/Industrial_and_Scientific_downstream/stage_inputs.json`。

RFSQ：3 层，每层 8 维，标量等级 `[4,4,4,2,2,2,2,2]`，每层隐式容量 2,048；不将它称为学习到的中心码本。当前实验使用标准化 embedding、`latent_activation=none` 和每层量化前 input clamp。碰撞微调保留重建项并加入编码空间 margin 损失；原始 SID 与末层碰撞修复后的 SID 分开核验，避免把修复后的唯一率当作原始量化唯一率。

## 环境

已验证的本地环境：Apple M5 Max、64 GiB 统一内存，Python 3.14、PyTorch MPS。`requirements-mps.txt` 记录本次环境的直接依赖版本；原 `requirements.txt` 是 CUDA 工作流，不宜在 Mac 上整份安装。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mps.txt
```

下载 Base 模型至 `models/Qwen2.5-1.5B/`；准备外部 embedding/行为源数据后，检查各数据准备脚本的 `--help` 与路径参数：

- `data/prepare_external_industrial_sid.py`：外部 embedding、ID 与交互适配。
- `data/build_industrial_downstream_inputs.py`：下游 CSV、商品映射、tokenizer 等输入。
- `data/tokenize_industrial_cpt.py`：纯 SID 序列及标签。
- `data/prepare_industrial_sft.py`、`data/prepare_industrial_rl.py`：原项目任务加载器生成的本地缓存。

不要在没有本机数据和已核验 SID 的情况下直接启动训练；部分实验/监控脚本默认指向这次运行的目录名。

## CPT

入口 `cpt_sid_sequence.py`：纯 SID 因果建模，不使用 `cpt.py` 中的辅助 prefix-ranking 目标。每个商品 3 个 SID token，用户序列直接拼接，训练标签按因果位置移位。训练前扩展词表，5,954 个新 SID token 为单独 token；模型词表最终为 157,619，输入输出 embedding 绑定。

本次：1 epoch、有效 batch 64、学习率 5e-5。恢复训练通过 `--resume_from_checkpoint`；切换 microbatch 时可用 `--sampler_batch_size` 保持数据遍历顺序。FP32 参数/AdamW 状态、BF16 autocast、梯度检查点。实际曾发生内存不足并恢复，已弃用的更新不计入最终训练曲线。

历史运行按 512 条验证样本的下一商品 SID loss 选择第 3,750 步检查点；训练目标本身仍是整条序列的因果建模。全量验证整序列 loss 约 1.2179。不要混淆目标商品和整序列的 loss 口径。绘图入口 `tools/plot_cpt_losses.py`。

## SFT

入口 `sft_mps.py`，从 CPT 导出模型开始，保留原项目三类任务和回答掩码：

| 任务 | 样本数 |
|---|---:|
| 历史 SID → 下一商品 SID | 259,983 |
| 商品 SID ↔ 标题 | 51,595 |
| 历史 SID → 下一商品标题 | 259,983 |
| 合计 | 571,561 |

只对回答计算 loss，提示词/padding 使用 `-100`。选择回答位置的 logits，损失及梯度已通过与完整 logits 实现的一致性测试。三个任务按回答 token 聚合，因此标题任务监督量更大；不要将混合训练 loss 直接与单一推荐任务验证 loss 比较。

本次：1 epoch，microbatch 8、累积 16、有效 batch 128，学习率 5e-5，4,466 次更新；最佳模型为第 3,500 步。全量 50,985 条验证回答 loss **2.243356**。训练/验证差距诊断脚本为 `tools/diagnose_sft_loss.py`。

## RL 与自动衔接

入口 `rl_mps.py`：单设备 MPS GRPO 适配，不是直接运行 CUDA/TRL 脚本。`ranking_grpo.py` 对应原 `rl.py` 的 `reward_type="ranking"`，即 exact-match 奖励加 rank-dependent 奖励，按完整生成组标准化，再分块反传。不是 `ranking_only`。

- 固定 SFT 参考模型；不使用参考同步、DAPO/GSPO、动态采样或真实目标注入。
- 每提示 16 个合法 SID beam 候选，长度惩罚 0；候选包含三层 SID、换行和 EOS。
- 有效更新批量为 8 个提示 × 16 候选 = 128 条生成序列。
- `micro_batch_size` 是反向传播的候选序列数；`rollout_batch_size` 是同时生成的提示数，不能混为一谈。
- 本次实测选择 microbatch 16、rollout 4、梯度累积 8；MPS 硬上限 44 GiB，调优候选峰值须 ≤40 GiB。
- 当前默认 1 epoch、学习率 1e-6、beta 0.04、余弦调度；AdamW 替代 CUDA paged optimizer。这些不是原多卡 `rl.sh` 的整套超参数。
- 原加载器生成训练提示 320,038 条，包含 SID 推荐、标题/描述识别、10,000 条标题历史推荐。奖励直接使用每行目标，避免原历史字典键冲突覆盖目标。

`tools/run_sft_to_rl.py` 每 30 秒等待 SFT 成功完成、全量验证、导出模型并释放 GPU，之后执行检查点清理、RL 内存调优和真实奖励 smoke，再启动正式 RL。失败/暂停不自动越过；不是按时钟猜测 SFT 结束。`tools/cleanup_completed_sft.py` 校验最终模型、词表、哈希及完整验证结果后，只删除已被取代的 best/last 检查点，保留最终模型和日志。

暂停当前 RL 并在完整更新后保存：

```bash
touch output/Industrial_RL_ranking_mps/PAUSE
```

本次 RL 在第 1,551 / 40,005 步按用户要求暂停，不能将它称为完成的 RL 实验。恢复时指定新的输出目录和已保存 checkpoint；不要重启自动控制器来覆盖旧运行目录。

## Base / CPT / SFT 全量测试（2026-10-16）

同一份 **50,985 条测试样本**，25,847 件商品候选目录，无负采样或已见商品过滤，合法 SID 约束的 10-beam 搜索（近似搜索，不是逐商品穷举打分）。每条仅一个正确目标，HitRate@K 等于 Recall@K。

| 模型 | HitRate@3 | HitRate@5 | HitRate@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Base＋SID 词表，未微调 | 0.0078% | 0.0275% | 0.0569% | 0.000052 | 0.000133 | 0.000227 |
| CPT | 0.9356% | 1.2710% | 1.7260% | 0.007264 | 0.008646 | 0.010110 |
| SFT | 1.0650% | 1.3788% | 1.9104% | 0.008407 | 0.009699 | 0.011413 |

Base 的新 SID embedding 按种子 42 初始化，未学到商品语义。Base/CPT 使用完整纯 SID 历史，预测三个 SID；SFT 使用与训练一致的指令模板和上下文截断，预测 SID＋换行/EOS。因此是对应输入形式的端到端比较，不是只改变权重的严格对照实验；不能直接解释为训练阶段的纯因果增益。SFT 的绝对命中率仍然较低。测试结果不用于后续选模或调参。

评估入口：

```bash
.venv/bin/python tools/run_recommendation_comparison.py
.venv/bin/python tools/watch_recommendation_comparison.py
```

`evaluate_stage_comparison.py` 保存逐条预测、数据校验信息、协议、进度及汇总表；中断可复用同模型/协议的已有预测。本次输出位于忽略目录 `output/Industrial_Base_CPT_SFT_test/`；RL 未参与这张表。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests
```

CPU 测试覆盖奖励与原实现等价、GRPO 分块梯度、合法 SID beam、回答掩码和选择性 logits 等价、指标边界以及安全清理。CPU 单测通过不代表所有 MPS 内存场景均已覆盖；内存测试和真实训练仍需本机数据/模型。
