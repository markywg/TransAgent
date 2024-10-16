# sh extractor_pixart_inall.sh DEVICE
DATASET=imagenet
DEVICE=$1

DATA="/path/to/dataset/folder"
OUTPUT='pixart_knowledge_inall'
SHOTS=16
SUB=all
MODAL=cross

for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=${DEVICE} python extractor_pixart.py \
    --split train \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file ../configs/datasets/${DATASET}.yaml \
    --config-file ../configs/knowledge_extraction/extractor_pixart.yaml \
    --output-dir ${OUTPUT} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    MODEL.MODAL ${MODAL}
done