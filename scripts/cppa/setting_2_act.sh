#!/bin/bash

set -e

source_domains=$1
target_domains=$2

echo "Training from $source_domains to $target_domains begins."

output_dir="./output/domain_adaptation/${source_domains}_${target_domains}"
dataset_config_file="./configs/datasets/DARS_${source_domains}_${target_domains}_TWO.yaml"
if [ -f "$dataset_config_file" ]; then
    echo "Dataset config file exists. Continuing..."
else
    dataset_config_file="./configs/datasets/DARS_${target_domains}_${source_domains}_TWO.yaml"
    if [ -f "$dataset_config_file" ]; then
        echo "Swapped source_domains and target_domains. Dataset config file exists. Continuing..."
    else
        echo "Dataset config file not found. Exiting..."
        exit 1
    fi
fi

config_file="./configs/trainers/CPPA/vit_b16.yaml"
echo "Starting training..."

# Check if output directory exists
if [ ! -d "$output_dir" ]; then
    echo "Output directory does not exist. Starting training..."
else
    # Check log status
    if [ -f "$output_dir/log.txt" ]; then
        target_line=$(tail -n 1 "$output_dir/log.txt")
        if ! echo "$target_line" | grep -q "Elapsed"; then
            echo "Log file incomplete. Cleaning output directory..."
            rm -rf "$output_dir"
        else
            echo "Training already completed. Skipping."
            exit 0
        fi
    else
        echo "Log file missing. Cleaning output directory..."
        rm -rf "$output_dir"
    fi
fi

export CUDA_VISIBLE_DEVICES=0

python ./train.py --root ./data --seed 1 --trainer CPPA \
    --config-file "$config_file" \
    --dataset-config-file "$dataset_config_file" \
    --output-dir "$output_dir" \
    --source-domains "$source_domains" \
    --target-domains "$target_domains" \
    --opts TRAINER.CPPA.N_CTX 16 \
    TRAINER.CPPA.PROMPT_DEPTH 9 \
    TRAINER.CPPA.FUSING mean \
    TRAINER.CPPA.PS True \
    DATASET.SUBSAMPLE_CLASSES all \
    DATASET.NUM_SHOTS 12 \
    TRAINER.CPPA.CTX_INIT ""
wait

echo "Training completed."
echo "Task from $source_domains to $target_domains finished."