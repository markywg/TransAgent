#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=TransAgent

DATASET=$1  # [caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft dtd eurosat ucf101 sun397 imagenet]
CFG=vit_b16_c2_ep50_batch4_4+4ctx_few_shot
SHOTS=$2
DEVICE=$3


for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo " The results exist at ${DIR}"
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
        TRAINER.MODAL fewshot
    fi
done
