# sh extractor_pixart_fewshot.sh DEVICE

DEVICE=$1

DATA="/path/to/dataset/folder"
OUTPUT='pixart_knowledge'
SUB=all
MODAL=fewshot


for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft dtd eurosat ucf101 sun397 imagenet
do
    for SHOTS in 1 2 4 8 16
    do
        for SEED in 1 2 3
        do
            DIR=${OUTPUT}_${SHOTS}shots_seed${SEED}
            CUDA_VISIBLE_DEVICES=${DEVICE} python extractor_pixart.py \
            --split train \
            --root ${DATA} \
            --seed ${SEED} \
            --dataset-config-file ../configs/datasets/${DATASET}.yaml \
            --config-file ../configs/knowledge_extraction/extractor_pixart.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB} \
            MODEL.MODAL ${MODAL}
        done
    done
done