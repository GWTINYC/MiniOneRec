#!/bin/bash

# MiniOneRec SFT 训练脚本 - 笔记本优化版本
# 针对单 GPU 笔记本电脑优化（8GB 显存）

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    # 【关键优化】使用 python 直接运行（1 GPU），而不是 torchrun（多 GPU）
    python sft.py \
            --base_model ./models/qwen3-1.7b \
            --batch_size 32 \
            --micro_batch_size 4 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/sft_model_laptop \
            --wandb_project wandb_proj \
            --wandb_run_name sft_laptop \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
            --freeze_LLM True
done
