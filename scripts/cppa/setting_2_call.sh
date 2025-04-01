#!/bin/bash

set -e

datasets=("UCM" "AID" "UCM" "NWPU" "AID" "NWPU")

for ((i=0; i<${#datasets[@]}; i+=2)); do
    source_dataset=${datasets[i]}
    target_dataset=${datasets[i+1]}
    bash ./setting_2_act_1_gpu.sh "$source_dataset" "$target_dataset"
    wait
    bash ./setting_2_act_1_gpu.sh "$target_dataset" "$source_dataset"
    wait
done

echo "Experiment completed."
echo "All tasks done!"