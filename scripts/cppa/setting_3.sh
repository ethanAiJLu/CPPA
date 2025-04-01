#!/bin/bash

set -e

source_domains=("UCM" "WHU" "RSSCN7" "AID")

for ((i=0; i<${#source_domains[@]}; i++)); do
    for ((j=0; j<${#source_domains[@]}; j++)); do
        if [ $i -ne $j ]; then
            source_domain=${source_domains[$i]}
            target_domain=${source_domains[$j]}

            echo "Training from $source_domain to $target_domain begins."

            output_dir="./output/domain_adaptation/four_domains/${source_domain}_${target_domain}"
            dataset_config_file="./configs/datasets/DARS_UCM_AID_WHU_RSSCN7_TWO.yaml"
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
                        continue
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

            echo "Training completed."
        fi
    done
done

echo "All tasks finished."