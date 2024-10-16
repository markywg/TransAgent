#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=TransAgent

DATASET=$1  # [caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft dtd eurosat ucf101 sun397 imagenet]
SEED=$2
DEVICE=$3

CFG=vit_b16_c2_ep20_batch4_4+4ctx
SHOTS=16
SUB=base
OUTPUT=output

DIR=${OUTPUT}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.MODAL base2novel
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.MODAL base2novel
fi