#!/bin/bash

# MiniOneRec 快速启动脚本
# 用法: bash quick_start.sh

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     MiniOneRec 快速启动检查清单                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. 环境激活
echo "1️⃣  激活 uv 虚拟环境..."
cd "$(dirname "$0")"
source .venv/bin/activate
echo "✅ 虚拟环境已激活"
echo ""

# 2. 环境验证
echo "2️⃣  验证环境信息..."
echo "   Python 版本: $(python --version)"
echo "   Python 位置: $(which python)"
echo "   虚拟环境: $VIRTUAL_ENV"
echo ""

# 3. 检查关键包
echo "3️⃣  检查关键包..."
python -c "
import torch
import transformers
import lightning
print(f'   ✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'   ✅ Transformers {transformers.__version__}')
print(f'   ✅ Lightning Utilities 已安装')
" 2>/dev/null || echo "   ⚠️  某些包加载有警告，但可继续"
echo ""

# 4. 数据检查
echo "4️⃣  检查数据文件..."
if [ -f "./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" ]; then
    echo "   ✅ 训练数据存在"
else
    echo "   ⚠️  训练数据未找到"
fi

if [ -f "./data/Amazon/index/Industrial_and_Scientific.index.json" ]; then
    echo "   ✅ SID 索引存在"
else
    echo "   ⚠️  SID 索引未找到"
fi

if [ -f "./data/Amazon/index/Industrial_and_Scientific.item.json" ]; then
    echo "   ✅ 商品元数据存在"
else
    echo "   ⚠️  商品元数据未找到"
fi
echo ""

# 5. 配置文件检查
echo "5️⃣  检查配置文件..."
[ -f "./config/zero2_opt.yaml" ] && echo "   ✅ DeepSpeed 配置存在" || echo "   ⚠️  配置文件未找到"
[ -f "./sft.sh" ] && echo "   ✅ SFT 脚本存在" || echo "   ⚠️  SFT 脚本未找到"
[ -f "./rl.sh" ] && echo "   ✅ RL 脚本存在" || echo "   ⚠️  RL 脚本未找到"
echo ""

# 6. GPU 检查
echo "6️⃣  GPU 信息..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "   ✅ 检测到 $GPU_COUNT 个 GPU"
    echo "   GPU 详情:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/      /'
else
    echo "   ⚠️  nvidia-smi 不可用（CPU 模式）"
fi
echo ""

# 7. 打印下一步指示
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              🚀 下一步操作                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "准备工作："
echo "  1. 编辑 sft.sh，修改 --base_model 为实际模型路径"
echo "  2. 编辑 sft.sh，修改 --output_dir 为期望的输出目录"
echo ""
echo "运行 SFT 训练："
echo "  source .venv/bin/activate"
echo "  bash sft.sh"
echo ""
echo "运行 RL 训练（在 SFT 完成后）："
echo "  source .venv/bin/activate"
echo "  bash rl.sh"
echo ""
echo "进行评估："
echo "  source .venv/bin/activate"
echo "  bash evaluate.sh"
echo ""
echo "📖 查看完整指南："
echo "  cat PROJECT_RUNGUIDE.md"
echo ""
