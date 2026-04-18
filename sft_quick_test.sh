#!/bin/bash

# MiniOneRec 快速体验脚本 - 5-10 分钟内完成
# 用极小数据集体验完整训练流程

export NCCL_IB_DISABLE=1

# 创建小数据集（仅 400 条数据 = 原始的 0.1%）
echo "【第一步】准备小数据集..."
head -400 ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv > /tmp/tiny_train.csv
head -100 ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv > /tmp/tiny_valid.csv

echo "训练数据行数："
wc -l /tmp/tiny_train.csv /tmp/tiny_valid.csv

category="Industrial_and_Scientific"

echo ""
echo "【第二步】开始快速训练（预计 5-10 分钟）..."

# 【快速优化参数】
# - batch_size: 32 → 8（减少 VRAM，加快前向传播）
# - max_len: 512 → 256（减少序列长度）
# - num_train_epochs: 1（只训练一轮）
# - eval_steps: 很大（跳过验证，节省时间）
# - logging_steps: 大（减少日志开销）

python sft.py \
    --base_model ./models/qwen3-1.7b \
    --batch_size 8 \
    --micro_batch_size 2 \
    --train_file /tmp/tiny_train.csv \
    --eval_file /tmp/tiny_valid.csv \
    --output_dir output_dir/sft_quick_test \
    --wandb_project wandb_proj \
    --wandb_run_name sft_quick_test \
    --category ${category} \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --freeze_LLM True \
    --cutoff_len 256 \
    --num_train_epochs 1 \
    --eval_steps 10000 \
    --logging_steps 10 \
    --save_steps 10000 \
    --gradient_accumulation_steps 1

echo ""
echo "✅ 快速训练完成！"
echo "模型保存位置: output_dir/sft_quick_test"
echo ""
echo "【后续可选】"
echo "1. 如果想评估效果，运行："
echo "   python evaluate.py --base_model output_dir/sft_quick_test --info_file ./data/Amazon/info/Industrial_and_Scientific.txt --category Industrial_and_Scientific --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv --result_json_data ./quick_test_result.json --batch_size 4 --num_beams 20 --max_new_tokens 256"
echo ""
echo "2. 体验完毕后，再用完整数据进行真正训练"
