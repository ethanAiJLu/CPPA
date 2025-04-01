#!/bin/bash

set -e

# Define source and target domains
source_domains="WHU"
target_domains="UCM"

echo "Training from source domain $source_domains to target domain $target_domains."

# Set up output directory and configuration files
output_dir="./output/${source_domains}_${target_domains}"
dataset_config_file="./configs/datasets/DARS_${source_domains}_${target_domains}_TWO.yaml"

# Check dataset config file existence
if [ ! -f "$dataset_config_file" ]; then
    dataset_config_file="./configs/datasets/DARS_${target_domains}_${source_domains}_TWO.yaml"
    if [ ! -f "$dataset_config_file" ]; then
        echo "Dataset configuration file not found. Exiting..."
        exit 1
    fi
fi

config_file="./configs/trainers/CPPA/vit_b16.yaml"

echo "Initiating training process..."

# Existing training completion check logic
if [ -d "$output_dir" ]; then
    if [ -f "$output_dir/log.txt" ]; then
        if grep -q "Elapsed" "$output_dir/log.txt"; then
            echo "Training already completed. Skipping."
            exit 0
        else
            rm -rf "$output_dir"
        fi
    fi
fi

export CUDA_VISIBLE_DEVICES=0

# Training command with generalized paths
python train.py --root ./data --seed 1 --trainer CPPA \
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
    DATASET.NUM_SHOTS 8 \
    TRAINER.CPPA.CTX_INIT ""

wait

echo "Training completed for task from $source_domains to $target_domains."