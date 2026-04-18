#!/bin/bash

# MiniOneRec RL 训练脚本 - 笔记本优化版本
# 针对单 GPU 笔记本电脑优化（8GB 显存）

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    # 【关键优化】直接使用 python 运行（1 GPU），而不是 accelerate launch（多 GPU）
    python rl.py \
                        --model_path output_dir/sft_model_laptop \
                        --train_batch_size 8 \
                        --eval_batch_size 16 \
                        --num_train_epochs 1 \
                        --gradient_accumulation_steps 4 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.5 \
                        --reward_type ranking \
                        --num_generations 4 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model False \
                        --beam_search False \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 5e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir output_dir/rl_model_laptop \
                        --wandb_run_name rl_laptop \
                        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
