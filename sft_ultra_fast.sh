#!/bin/bash

# MiniOneRec 極速體驗腳本 - 5 分鐘內完成
# 用極小數據集 + 極限優化參數

export NCCL_IB_DISABLE=1

# 創建超小數據集（僅 20 條數據 = 原始的 0.055%）
echo "【第一步】準備超小數據集..."
head -20 ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv > /tmp/tiny_train.csv
head -10 ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv > /tmp/tiny_valid.csv

echo "訓練數據行數："
wc -l /tmp/tiny_train.csv /tmp/tiny_valid.csv

category="Industrial_and_Scientific"

echo ""
echo "【第二步】極速訓練（預計 3-5 分鐘）..."

# 【極限優化參數】
# - batch_size: 8 → 1（單樣本批次，最小開銷）
# - micro_batch_size: 2 → 1（無梯度累積）
# - cutoff_len: 256 → 128（最小序列）
# - num_train_epochs: 1（單輪）
# - max_steps: 10（只訓練 10 步，可快速驗證）
# - logging_steps: 999999（完全禁用日誌）
# - eval_steps: 999999（完全禁用驗證）
# - save_steps: 999999（完全禁用中間保存）

python sft.py \
    --base_model ./models/qwen3-1.7b \
    --batch_size 1 \
    --micro_batch_size 1 \
    --train_file /tmp/tiny_train.csv \
    --eval_file /tmp/tiny_valid.csv \
    --output_dir output_dir/sft_ultra_fast \
    --wandb_project wandb_proj \
    --wandb_run_name sft_ultra_fast \
    --category ${category} \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --freeze_LLM True \
    --cutoff_len 128 \
    --num_train_epochs 1 \
    --eval_steps 999999 \
    --logging_steps 999999 \
    --save_steps 999999 \
    --gradient_accumulation_steps 1

echo ""
echo "✅ 極速訓練完成！"
echo "模型保存位置: output_dir/sft_ultra_fast"
echo ""
echo "【說明】"
echo "- 此配置用於快速驗證流程，不涉及實際訓練效果"
echo "- 訓練數據: 20 條 (0.055% 原始數據)"
echo "- 序列長度: 128 (25% 原始)"
echo "- 無日誌、無驗證、無檢查點保存"
echo ""
echo "【測試效果】"
echo "python evaluate.py --base_model output_dir/sft_ultra_fast --info_file ./data/Amazon/info/Industrial_and_Scientific.txt --category Industrial_and_Scientific --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv --result_json_data ./ultra_fast_result.json --batch_size 2 --num_beams 10 --max_new_tokens 128"
