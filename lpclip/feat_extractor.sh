# sh feat_extractor.sh
DATA="/path/to/dataset"
OUTPUT='./clip_feat/'
SEED=1
DEVICE=$1

for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
do
    for SPLIT in train val test
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --config-file ../configs/trainers/CoOp/vit_b16_val.yaml \
        --output-dir ${OUTPUT} \
        --eval-only
    done
done
