# sh extractor_shikra.sh DEVICE

DEVICE=$1

DATA="/path/to/dataset/folder"
OUTPUT='shikra_knowledge'
SHOTS=16
SUB=base

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft dtd eurosat ucf101 sun397 imagenet
do
    for SEED in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} python extractor_shikra.py \
        --split train \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --config-file ../configs/knowledge_extraction/extractor_shikra.yaml \
        --output-dir ${OUTPUT} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    done
done
