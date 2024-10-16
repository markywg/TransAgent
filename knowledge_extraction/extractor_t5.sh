# sh extractor_t5.sh

DATA="/path/to/dataset/folder"
OUTPUT='t5_text_feat_wmask'
PRETRAINED='/path/to/PixArt-alpha'

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft dtd eurosat ucf101 sun397 imagenet
do
    CUDA_VISIBLE_DEVICES=0 python extractor_t5.py \
    --root ${DATA} \
    --dataset-config-file ../configs/datasets/${DATASET}.yaml \
    --output-dir ${OUTPUT} \
    --pretrained_models_dir ${PRETRAINED}
done