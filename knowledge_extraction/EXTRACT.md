# Knowledge Extraction for Agent Collaboration

### Above All
Choose a directory ```$CKPT``` where you store all the model checkpoints needed for the experiments.
<br>
Remember to replace all the checkpoint path with ```$CKPT```/checkpoints in the config files.

### Vision Agent

Download the model checkpoints from the following urls and put them under ```$CKPT```. Alter the path at line 602-621 in [transagent.py](../trainers/transagent.py).
<br>
- [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)
- [DINO](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)
- [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [ViTDet](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl)

For SAM, since we only need the image backbone, run [extract_sam_vit.py](extract_sam_vit.py) first to extract the backbone checkpoint. And replace the path with the extracted checkpoint.
<br>
The knowledge from vision agents are extracted on the run, as shown at line 652-655 in [transagent.py](../trainers/transagent.py).

### Language Agent

The knowledge from language agents are extracted in advance and stored under the project directory.
- Vicuna: [template](../template)
- GPT3: [template_protext](../template_protext)

### Multi-Modal Agent

#### (1) Stable Diffusion

First, prepare the environment following the [official codebase](https://github.com/CompVis/stable-diffusion), download the model checkpoint from [huggingface](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main) and put it under ```$CKPT```.
<br>
Next, change the ```CONFIG``` and ```CKPT``` in [extractor_sd.yaml](../configs/knowledge_extraction/extractor_sd.yaml) and ```DATA``` in [extractor_sd.sh](extractor_sd.sh).
<br>
Then, simply run the following command to extract the knowledge from SD for base-to-novel experiments:
```
sh extractor_sd.sh {DEVICE}
```
The extracted knowledge will be stored in each dataset folder under the 'sd_knowledge' directory.
Similarly, run [extractor_sd_fewshot.sh](extractor_sd_fewshot.sh) and [extractor_sd_inall.sh](extractor_sd_inall.sh)
to extract the knowledge for few-shot and cross-dataset experiments.
<br>
The knowledge extraction pipeline is in the [file](extractor_sd.py).

#### (2) Pixart

First, prepare the environment following the [official codebase](https://github.com/PixArt-alpha/PixArt-alpha/tree/53dac066f60fe5fdbdde4f0360145ca96d4cc38c), download the model checkpoint from [huggingface](https://huggingface.co/PixArt-alpha/PixArt-alpha) and put it under ```$CKPT```.
Change line 1-2, 15-16 in [PixArt_xl2_img512.py](../configs/pixart/PixArt_xl2_img512.py).
<br>
For Pixart, since the T5 text encoder is too large, we need to pre-extract the text features before extracting the knowledge from the image backbone.
<br>
For base-to-novel experiments, change the ```DATA``` and ```PRETRAINED``` in [extractor_t5_base.sh](extractor_t5_base.sh) and run the following command to extract the T5 text features:
```
sh extractor_t5_base.sh
```
Similarly, use [extractor_t5.sh](extractor_t5.sh) to extract the T5 text features for cross-dataset and few-shot experiments.
<br>
Next, change the ```CONFIG``` in [extractor_pixart.yaml](../configs/knowledge_extraction/extractor_pixart.yaml) and ```DATA``` in [extractor_pixart.sh](extractor_pixart.sh).
<br>
Then, simply run the following command to extract the knowledge from Pixart for base-to-novel experiments:
```
sh extractor_sd.sh {DEVICE}
```
The extracted knowledge will be stored in each dataset folder under the 'pixart_knowledge' directory.
Similarly, run [extractor_pixart_fewshot.sh](extractor_pixart_fewshot.sh) and [extractor_pixart_inall.sh](extractor_pixart_inall.sh)
to extract the knowledge for few-shot and cross-dataset experiments.
<br>
The knowledge extraction pipeline is in the [file](extractor_pixart.py).

#### (3) BLIP2

First, download model checkpoint from [huggingface](https://huggingface.co/Salesforce/blip2-opt-2.7b) and put it under ```$CKPT```
<br>
Next, change the ```CKPT``` in [extractor_blip2.yaml](../configs/knowledge_extraction/extractor_blip2.yaml) and ```DATA``` in [extractor_blip2.sh](extractor_blip2.sh).
<br>
Then, simply run the following command to extract the knowledge from BLIP2 for base-to-novel experiments:
```
sh extractor_blip2.sh {DEVICE}
```
The extracted knowledge will be stored in each dataset folder under the 'blip2_knowledge' directory.
<br>
Similarly, run [extractor_blip2_fewshot.sh](extractor_blip2_fewshot.sh) and [extractor_blip2_inall.sh](extractor_blip2_inall.sh)
to extract the knowledge for few-shot and cross-dataset experiments.
<br>
The knowledge extraction pipeline is in the [file](extractor_blip2.py).

#### (4) Shikra

First, prepare the environment following the [official codebase](https://github.com/shikras/shikra), download model checkpoint and put it under ```$CKPT```.
Download CLIP-L/14 from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14) and put it under ```$CKPT``` as well.
<br>
Next, change the ```CONFIG``` in [extractor_shikra.yaml](../configs/knowledge_extraction/extractor_shikra.yaml) and ```DATA``` in [extractor_shikra.sh](extractor_shikra.sh).
Change line 7-8 in [shikra.py](../configs/shikra/shikra.py).
<br>
Then, simply run the following command to extract the knowledge from Shikra for base-to-novel experiments:
```
sh extractor_shikra.sh {DEVICE}
```
The extracted knowledge will be stored in each dataset folder under the 'shikra_knowledge' directory.
Similarly, run [extractor_shikra_fewshot.sh](extractor_shikra_fewshot.sh) and [extractor_shikra_inall.sh](extractor_shikra_inall.sh)
to extract the knowledge for few-shot and cross-dataset experiments.
<br>
The knowledge extraction pipeline is in the [file](extractor_shikra.py).