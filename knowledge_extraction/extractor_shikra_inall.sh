# sh extractor_shikra_inall.sh DEVICE
DATASET=imagenet
DEVICE=$1

DATA="/path/to/dataset/folder"
OUTPUT='shikra_knowledge_inall'
SHOTS=16
SUB=all

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