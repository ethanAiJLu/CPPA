#!/bin/bash

set -e

# Remove error notification trap
trap 'echo "Error occurred. Exiting..."; exit 1' ERR

# VisDA-2017 has only one source-target pair
source_domain="train"
target_domain="validation"

echo "Now the source domain is $source_domain, and the target domain is $target_domain. Begin to train."

# Update paths to use relative paths and CPPA
output_dir="./output/visda2017/${source_domain}_${target_domain}"
dataset_config_file="./configs/datasets/visda2017_CPPA.yaml"
config_file="./configs/trainers/CPPA/vit_b16.yaml"

echo "Start training..."

# Check if output_dir exists
if [ ! -d "$output_dir" ]; then
    echo "Output directory does not exist. Starting training..."
else
    # Check if log.txt exists and contains "Elapsed" in the last line
    if [ -f "$output_dir/log.txt" ]; then
        target_line=$(tail -n 1 "$output_dir/log.txt")
        if ! echo "$target_line" | grep -q "Elapsed"; then
            echo "Log file does not contain 'Elapsed'. Deleting output directory..."
            rm -rf "$output_dir"
            echo "Output directory deleted. Starting training..."
        else
            echo "+==================================================================+"
            echo "Log file contains 'Elapsed'."
            echo "Training already completed. Skipping."
            echo "+==================================================================+"
            exit 0  # Exit if training already completed
        fi
    else
        echo "Log file does not exist. Starting training..."
        rm -rf "$output_dir"
        echo "Output directory deleted. Starting training..."
    fi
fi

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Run training command
python ./train.py --root ./data --seed 1 --trainer CPPA \
    --config-file "$config_file" \
    --dataset-config-file "$dataset_config_file" \
    --output-dir "$output_dir" \
    --source-domains "$source_domain" \
    --target-domains "$target_domain" \
    --opts TRAINER.CPPA.N_CTX 16 \
    TRAINER.CPPA.PROMPT_DEPTH 9 \
    TRAINER.CPPA.FUSING mean \
    TRAINER.CPPA.PS True \
    DATASET.SUBSAMPLE_CLASSES all \
    DATASET.NUM_SHOTS 8 \
    TRAINER.CPPA.CTX_INIT ""

wait

echo "Finish training."
echo "All training tasks completed successfully!"