# TransAgent Training

We provide bash scripts in [scripts/](../scripts) for training TransAgent.
Make sure to update the `DATA` variable with dataset path in the script file and run the commands from the main directory `transagent/`.
Below we provide training and testing instructions for TransAgent.

### Before Training Launch
Before training TransAgent, follow instructions in [EXTRACT.md](../knowledge_extraction/EXTRACT.md) to extract knowledge from external agents.

### (1) Base-to-novel generalization setting
The base-to-novel TransAgent configuration is provided in config file at `configs/trainers/TransAgent/vit_b16_c2_ep20_batch4_4+4ctx.yaml`. All hyper-parameters such as VISION_LOSS_WEIGHT, LLM_LOSS_WEIGHT, MM_LOSS_WEIGHT, prompt length and prompt depth etc., can be modified using this config file.

Run the commands below to train TransAgent on ImageNet.

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# train and evaluate on base classes
bash scripts/transagent/base2new_train.sh imagenet 1 {DEVICE}
# evaluate on novel classes
bash scripts/transagent/base2new_test.sh imagenet 1 {DEVICE}

# seed=2
# train and evaluate on base classes
bash scripts/transagent/base2new_train.sh imagenet 2 {DEVICE}
# evaluate on novel classes
bash scripts/transagent/base2new_test.sh imagenet 2 {DEVICE}

# seed=3
# train and evaluate on base classes
bash scripts/transagent/base2new_train.sh imagenet 3 {DEVICE}
# evaluate on novel classes
bash scripts/transagent/base2new_test.sh imagenet 3 {DEVICE}
```
Please replace the `{DEVICE}` with the device you want to run the experiments on. 

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– TransAgent/
|   |   |   |   |   |–– vit_b16_c2_ep20_batch4_4+4ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– TransAgent/
|   |   |   |   |   |–– vit_b16_c2_ep20_batch4_4+4ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# print averaged results for base classes
python output/base2new/train_base/imagenet/shots_16/TransAgent/vit_b16_c2_ep20_batch4_4+4ctx --test-log
# print averaged results for novel classes
python output/base2new/test_new/imagenet/shots_16/TransAgent/vit_b16_c2_ep20_batch4_4+4ctx --test-log
```

The above steps can be repeated for other individual datasets.

### (2) Cross-Dataset Transfer setting
We provide instructions to train TransAgent on ImageNet using all 1000 classes with 16 shots and then evaluate it directly on new downstream datasets.
The corresponding cross-dataset config for TransAgent is available at: `configs/trainers/TransAgent/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets.yaml`.
* Firstly, train TransAgent on imagenet in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/transagent/xd_train.sh imagenet 1 {DEVICE}
# seed=2 
bash scripts/transagent/xd_train.sh imagenet 2 {DEVICE}
# seed=3 
bash scripts/transagent/xd_train.sh imagenet 3 {DEVICE}
```

* Now directly evaluate the ImageNet trained model on downstream cross-datasets.

```bash
# Other possible dataset values includes [food101, dtd, ucf101, oxford_flowers, fgvc_aircraft, sun397, eurosat]

for SEED in 1 2 3
do
    bash scripts/transagent/xd_test.sh caltech101 ${SEED} {DEVICE}
    bash scripts/transagent/xd_test.sh oxford_pets ${SEED} {DEVICE}
    bash scripts/transagent/xd_test.sh stanford_cars ${SEED} {DEVICE}
done
```
You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.


### (3) Domain Generalization setting
We use the same ImageNet trained TransAgent model for domain generalization experiments. The steps are similar to the above cross-dataset experiments, however, the trained model is now evaluated on ImageNet variants.
The corresponding domain generalization config for TransAgent is available at: `configs/trainers/TransAgent/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets.yaml`.
* Evaluate ImageNet trained model on different variants of ImageNet (datasets with domain shifts).

```bash
for SEED in 1 2 3
do
    bash scripts/transagent/xd_test.sh imagenetv2 ${SEED} {DEVICE}
    bash scripts/transagent/xd_test.sh imagenet_sketch ${SEED} {DEVICE}
    bash scripts/transagent/xd_test.sh imagenet_a ${SEED} {DEVICE}
    bash scripts/transagent/xd_test.sh imagenet_r ${SEED} {DEVICE}
done
```


You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.

### (4) Few-shot setting 
In this setting, TransAgent is trained using all classes on individual datasets with different few-shot splits (K = 1, 2, 4, 8, 16). The corresponding few-shot setting config for TransAgent is available at: `configs/trainers/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot.yaml`.

Now use the training script `scripts/transagent/few_shot.sh` and run the commands below to calculate the results on imagenet dataset with different shots over 3 seeds:

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# train and test on given dataset for K=1 shot
bash scripts/transagent/few_shot.sh imagenet 1 {DEVICE}
# train and test on given dataset for K=2 shot
bash scripts/transagent/few_shot.sh imagenet 2 {DEVICE}
# train and test on given dataset for K=4 shot
bash scripts/transagent/few_shot.sh imagenet 4 {DEVICE}
# train and test on given dataset for K=8 shot
bash scripts/transagent/few_shot.sh imagenet 8 {DEVICE}
# train and test on given dataset for K=16 shot
bash scripts/transagent/few_shot.sh imagenet 16 {DEVICE}
```


You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.
<br>

Please use the corresponding config and script files and follow the same instructions as provided for TransAgent for training and testing. 
This repository also supports using official [CoOp](CoOp.md) and [Co-CoOp](Co-CoOp.md) configs and models.
