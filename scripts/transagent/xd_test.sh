#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=TransAgent

DATASET=$1
SEED=$2
DEVICE=$3

CFG=vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets
SHOTS=16
OUTPUT=output

DIR=${OUTPUT}/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${OUTPUT}/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --eval-only \
    TRAINER.MODAL cross
fi