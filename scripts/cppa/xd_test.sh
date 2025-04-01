#!/bin/bash
# custom config
DATA=/path/to/data
TRAINER=CPPA

DATASET=$1
SEED=$2

CFG=vit_b16
SHOTS=16

for SEED in 1 2 3
do
    DIR=output/crossdata/${DATASET}/${SHOTS}shots/seed${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/${DATASET}/16shots/seed${SEED}\
    --load-epoch 5 \
    --eval-only
done