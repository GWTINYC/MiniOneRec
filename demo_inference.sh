#!/bin/bash

# MiniOneRec 秒级演示脚本 - 真正 1 分钟完成
# 不做任何训练，只演示推理流程

export NCCL_IB_DISABLE=1

echo "【秒级演示】跳过训练，直接用预训练模型做推理"
echo ""
echo "方案 1: 用原始预训练模型（无训练）"
echo "  优点: 立即可用，秒速完成"
echo "  用途: 验证推理流程"
echo ""

category="Industrial_and_Scientific"

# 查找完整的文件路径
info_file=$(ls -f ./data/Amazon/info/${category}*.txt | head -1)

echo "【开始推理演示】"
echo "用预训练模型: ./models/qwen3-1.7b"
echo "测试集样本: 100"
echo "信息文件: $info_file"
echo ""

# 创建小测试集（仅 100 条用于快速演示）
test_file=$(ls -f ./data/Amazon/test/${category}*.csv | head -1)
head -100 "$test_file" > /tmp/test_demo.csv

python evaluate.py \
    --base_model ./models/qwen3-1.7b \
    --info_file "$info_file" \
    --category ${category} \
    --test_data_path /tmp/test_demo.csv \
    --result_json_data ./demo_result.json \
    --batch_size 2 \
    --num_beams 10 \
    --max_new_tokens 128

echo ""
echo "【计算评估指标】"
python calc.py --path ./demo_result.json --item_path "$info_file"

echo ""
echo "✅ 演示完成！"
echo "预测结果已保存到: demo_result.json"
echo ""
echo "【接下来的选择】"
echo "1. 如果推理无误，开始真正训练:"
echo "   bash sft_ultra_fast.sh"
echo ""
echo "2. 或用快速训练脚本（3 小时）:"
echo "   bash sft_quick_test.sh"
echo ""
echo "3. 或用完整训练脚本（36+ 小时）:"
echo "   bash sft_laptop.sh"
