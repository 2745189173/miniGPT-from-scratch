# miniGPT-from-scratch

[English](README.md) | [简体中文](README.zh-CN.md)

一个使用 PyTorch 从底层实现的小型 Decoder-only Transformer 语言模型，涵盖
byte-level BPE、训练、断点续训、文本生成和受控实验。

这个仓库定位为教育型实现和实验平台，目标是让 GPT 的内部机制可以被阅读、验证
和修改。它不是生产级聊天机器人，也不能替代经过大规模通用语料预训练的语言模型。

## 当前版本

`v1.0.1` 完成了项目预定的学习范围、公开发布润色和中英文双语文档。最终模型在
Tiny Shakespeare 上训练，使用从零实现的 512-token byte-level BPE 词表。

## 核心亮点

- 从零实现确定性的字符 tokenizer 和 byte-level BPE tokenizer；
- 实现单头、多头因果自注意力；
- 使用 pre-norm Transformer Block、残差连接和 GELU FeedForward；
- 使用 YAML 配置驱动 AdamW 训练和定期验证；
- 保存最佳 checkpoint，并支持恢复模型、优化器、全局 step 和 RNG 状态；
- 支持 temperature 和 top-k 的自回归生成；
- 使用独立 run name 管理实验产物，并记录 12 组受控实验；
- 兼容旧 tokenizer、activation 和 weight tying checkpoint；
- 提供 focused checks、Ruff 代码检查和 GitHub Actions CI。

## 最终结果

| 指标 | 字符模型 E010 | BPE 模型 E012 |
|---|---:|---:|
| 词表大小 | 65 | 512 |
| 参数量 | 816,705 | 931,584 |
| 语料 token 数 | 1,115,394 | 568,210 |
| 64-token 上下文估算覆盖字符数 | 64.0 | 125.6 |
| 估算 bits per character | 2.4516 | 2.3374 |

BPE 让 token 数减少了 **49.06%**，让相同 token 长度下的上下文覆盖接近翻倍，
并使估算 bits per character 改善 **4.66%**。

最终 BPE checkpoint 的生成样本：

```text
they hands that can trumble with laughter,
Captain is the daughter, hearty of your swork;
Well I have duke me, and you are a maids the devil'd, and these a fear!

KING RICHARD II:
Now, go my lord, and can her give me business to his troge about bod,
```

## 模型架构

```text
UTF-8 文本 -> byte-level BPE -> token ids
-> token embedding + learned positional embedding
-> 4 x pre-norm Transformer blocks
   -> 4-head causal self-attention
   -> GELU FeedForward
-> final LayerNorm -> LM head -> next-token logits
```

最终配置：context 64、4 层、4 个 attention heads、embedding width 128、
dropout 0.1，共 931,584 个参数。

## 项目目标

这个项目从零实现并验证 GPT 风格语言模型的核心机制：

- 字符 tokenizer 和 byte-level BPE；
- token embedding 和位置编码；
- causal self-attention 和 multi-head attention；
- Transformer Block；
- next-token prediction；
- 训练、验证、最佳 checkpoint 和断点续训；
- 自回归生成和解码策略；
- tokenizer 与架构消融实验。

## 完成路线

- [x] 项目结构
- [x] 字符级 tokenizer
- [x] dataset 和 batch construction
- [x] 最小语言模型 baseline
- [x] causal self-attention
- [x] multi-head attention
- [x] Transformer Block
- [x] 训练循环
- [x] 验证 loss 和 checkpoint
- [x] 自回归生成
- [x] temperature 和 top-k
- [x] loss curve
- [x] 解码策略对比
- [x] byte-level BPE 核心
- [x] BPE 训练与生成全链路
- [x] 架构和 tokenizer 消融实验

v1.0 之后的可选方向包括 KV cache、RoPE、top-p、混合精度和更大规模训练。

## 环境安装

创建 Python 环境并安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

下载 Tiny Shakespeare：

```powershell
python scripts\download_data.py
```

根据 `configs/tiny.yaml` 训练确定性的 BPE-512 tokenizer：

```powershell
python scripts\train_tokenizer.py
```

tracked tokenizer 文件会生成在：

```text
artifacts/tokenizers/tiny_shakespeare_bpe_512.json
```

Tiny Shakespeare 下载自 Andrej Karpathy 的 `char-rnn` 仓库，来源和使用说明见
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md)。

## 验证

运行 focused checks：

```powershell
python scripts\check_data.py
python scripts\check_bpe.py
python scripts\check_tokenizer_integration.py
python scripts\check_attention.py
python scripts\check_block.py
python scripts\check_model.py
python scripts\check_gradients.py
```

运行代码规范检查：

```powershell
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
```

## 训练与生成

使用最终 BPE 配置训练：

```powershell
python scripts\train.py
```

单独复现字符模型 baseline：

```powershell
python scripts\train.py --config configs\char_baseline.yaml
```

绘制 loss curve 并从最佳 checkpoint 生成文本：

```powershell
python experiments\loss_curves.py
python scripts\generate.py
```

对比 tokenizer 和最终模型：

```powershell
python experiments\tokenizer_comparison.py
python experiments\final_comparison.py
```

checkpoint、loss history、PNG 和生成样本属于本地实验产物，默认不提交 Git。

## 仓库结构

```text
src/          模型、注意力、Block、tokenizer 和生成逻辑
scripts/      数据下载、tokenizer 训练、模型训练和检查脚本
configs/      最终 BPE 和字符 baseline 配置
experiments/  对比脚本、可视化脚本和实验日志
artifacts/    tracked BPE merge rules
report/       最终技术报告
```

## 项目资料

- [实验日志](experiments/EXPERIMENT_LOG.md)：E001-E012 的配置、指标和结论。
- [技术报告](report/technical_report.md)：最终架构、对比结果、限制和范围。

## 项目边界

这是一个可以完整训练、恢复、生成和做实验的 miniGPT，但不是通用大模型：

- 训练语料只有 Tiny Shakespeare；
- 参数量约 93 万；
- 生成内容主要模仿 Shakespeare 风格；
- 不具备通用问答、事实知识或指令遵循能力。

它的核心价值是完整实现和验证 GPT 的关键机制，而不是与生产级 LLM 比较能力。

## License

源码使用 [MIT License](LICENSE)。数据来源见
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md)。
