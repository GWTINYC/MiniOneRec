# MiniOneRec 项目运行指南

## 📋 项目概述

**MiniOneRec** 是开源的生成式推荐系统框架，包含三个训练阶段：

```
数据 → SID 构建 → SFT 训练 → RL 训练 → 推荐模型
```

## 🏗️ 项目层级

```
MiniOneRec/
├── sft.py / sft.sh              # SFT 阶段（第 1 步）
├── rl.py / rl.sh                # RL 阶段（第 2 步）  
├── minionerec_trainer.py        # 主训练器
├── data.py                      # 数据加载（核心）
├── evaluate.py                  # 评估指标
├── data/Amazon/                 # 数据目录
│   ├── train/                   # 训练集
│   ├── valid/                   # 验证集
│   ├── test/                    # 测试集
│   └── index/                   # SID 索引 + 商品元数据
├── rq/                          # 向量量化模块（SID 构建）
│   ├── text2emb/                # 文本→嵌入
│   ├── models/                  # RQ VAE 模型
│   └── trainer.py               # 量化训练器
└── config/                      # 配置文件
    └── zero2_opt.yaml           # DeepSpeed 配置
```

## 🚀 快速开始

### 1️⃣ 环境激活
```bash
cd /home/wxh/MiniOneRec
source .venv/bin/activate
python --version  # 验证 Python 3.10
```

### 2️⃣ 检查数据
```bash
ls -la data/Amazon/train/          # 确保训练数据存在
ls -la data/Amazon/index/          # 确保 SID 索引存在
```

### 3️⃣ SFT 训练（第 1 步最重要）

**编辑 `sft.sh`，修改关键参数：**

```bash
# 1. 设置基础模型路径（从 HuggingFace 下载）
--base_model /path/to/qwen-7b-instruct

# 2. 设置输出目录
--output_dir ./output/sft_model

# 3. 可选：设置 W&B 日志
--wandb_project "minionerec"
--wandb_run_name "sft_baseline"
```

**运行 SFT：**
```bash
bash sft.sh
# 或手动指定多 GPU
torchrun --nproc_per_node 8 sft.py \
    --base_model /path/to/model \
    --train_file ./data/Amazon/train/*.csv \
    --batch_size 1024 \
    --output_dir ./output/sft_model
```

**关键参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--base_model` | 基础 LLM 模型路径 | **必需** |
| `--micro_batch_size` | GPU 单个批次大小 | 16 |
| `--batch_size` | 累积批次大小 | 1024 |
| `--freeze_LLM` | 冻结 LLM 权重 | False |
| `--train_from_scratch` | 从零训练 | False |

**输出：**
- ✅ 模型文件：`./output/sft_model/`
- ✅ 日志：`./runs/` 或 W&B

### 4️⃣ RL 训练（第 2 步）

**编辑 `rl.sh`，修改关键参数：**

```bash
# 1. 使用 SFT 模型
--model_path ./output/sft_model

# 2. 设置输出目录
--output_dir ./output/rl_model

# 3. 训练参数
--num_train_epochs 2
--learning_rate 1e-5
--num_generations 16
```

**运行 RL：**
```bash
bash rl.sh
# 或手动指定
accelerate launch \
    --config_file ./config/zero2_opt.yaml \
    rl.py \
    --model_path ./output/sft_model \
    --train_file ./data/Amazon/train/*.csv \
    --num_train_epochs 2 \
    --output_dir ./output/rl_model
```

**关键参数说明：**
| 参数 | 说明 | 典型值 |
|------|------|------|
| `--model_path` | 需要 SFT 模型 | **必需** |
| `--num_generations` | 每个 prompt 生成候选数 | 16 |
| `--reward_type` | 奖励类型 | ranking |
| `--beam_search` | 使用束搜索 | True |
| `--beta` | KL 散度系数 | 1e-3 |
| `--temperature` | 采样温度 | 1.0 |

**输出：**
- ✅ 最终模型：`./output/rl_model/`
- ✅ 日志记录

### 5️⃣ 评估（可选）

```bash
python evaluate.py \
    --model_path ./output/rl_model \
    --test_file ./data/Amazon/test/*.csv \
    --batch_size 128

# 或使用脚本
bash evaluate.sh
```

**评估指标：**
- HR@K (Hit Rate)
- NDCG@K (Normalized Discounted Cumulative Gain)

---

## 📊 数据格式说明

### 训练数据（CSV 格式）
```csv
user_id,item_id,timestamp,category,title,description
user_001,item_123,1234567890,Electronics,Product Name,Product Description
```

### SID 索引文件 (`*.index.json`)
```json
{
  "item_001": [1, 24, 156],  // item → [token1, token2, token3]
  "item_002": [2, 45, 289]
}
```

### 商品元数据 (`*.item.json`)
```json
{
  "item_001": {
    "title": "Product Title",
    "description": "Product Description",
    "category": "Electronics"
  }
}
```

---

## ⚙️ 高级配置

### 多 GPU 训练

**SFT 阶段：** 使用 `torchrun`
```bash
torchrun --nproc_per_node 8 sft.py ...
```

**RL 阶段：** 使用 `accelerate` + DeepSpeed ZeRO-2
```bash
accelerate launch \
    --config_file ./config/zero2_opt.yaml \
    rl.py ...
```

### 内存不足解决方案

1. 降低 `--micro_batch_size`（SFT）：从 16 → 8 或 4
2. 增加 `--gradient_accumulation_steps`（RL）：保证有效批大小
3. 使用 `--freeze_LLM`：只训练 embedding，减少参数

### 分布式训练故障排除

```bash
# 禁用 InfiniBand（如果有网络问题）
export NCCL_IB_DISABLE=1

# 检查 GPU 连接
nvidia-smi

# 验证多进程：
python -c "import torch; print(torch.cuda.device_count())"
```

---

## 📝 项目核心文件说明

| 文件 | 功能 | 重要性 |
|------|------|-------|
| `minionerec_trainer.py` | GRPO 训练器（RL 核心） | ⭐⭐⭐ |
| `data.py` | 数据加载管道 | ⭐⭐⭐ |
| `evaluate.py` | 推荐指标计算 | ⭐⭐ |
| `LogitProcessor.py` | 约束解码（确保生成有效 token） | ⭐⭐ |
| `sft.py` | SFT 训练循环 | ⭐⭐ |
| `rl.py` | RL 训练循环 | ⭐⭐ |

---

## 🔍 常见问题

### Q: 如果没有 GPU 怎么办？
A: 修改 `sft.sh` 和 `rl.sh` 中的 `--nproc_per_node` 为 1，或使用 CPU（会很慢）。

### Q: 如何使用自己的数据？
A: 将数据转换为 CSV 格式，放在 `data/` 目录下，修改脚本中的 `--train_file` 路径。

### Q: 训练时间需要多长？
A: 
- SFT：通常 1-2 小时（8 个 A100 GPU）
- RL：通常 2-4 小时（8 个 A100 GPU）

### Q: 如何查看训练进度？
A: 
- 如果使用 W&B：访问 https://wandb.ai 查看实时日志
- 本地日志：`./runs/` 目录

---

## 💾 保存和恢复

### 恢复训练
```bash
# SFT 继续训练
torchrun --nproc_per_node 8 sft.py \
    --resume_from_checkpoint ./output/sft_model/checkpoint-500
```

### 推理使用已训练的模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./output/rl_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 生成推荐
```

---

## 📚 参考资源

- **论文**：https://arxiv.org/abs/2510.24431
- **HuggingFace 模型**：https://huggingface.co/kkknight/MiniOneRec
- **ModelScope**：https://modelscope.cn/models/k925238839/MiniOneRec

---

## 🛠️ 环境检查清单

- [ ] `source .venv/bin/activate` ✓
- [ ] `python --version` 显示 3.10 ✓
- [ ] `pip list | grep torch` 显示 PyTorch ✓
- [ ] `torch.cuda.is_available()` 返回 True ✓
- [ ] 数据文件存在于 `data/Amazon/` ✓
- [ ] 配置文件正确修改 ✓
- [ ] GPU 可用：`nvidia-smi` ✓

---

祝你训练顺利！ 🚀
