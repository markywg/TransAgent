feature_dir=clip_feat

#  OxfordPets OxfordFlowers FGVCAircraft DescribableTextures EuroSAT StanfordCars Food101 SUN397 Caltech101 UCF101 ImageNet
for DATASET in SUN397
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3
done
