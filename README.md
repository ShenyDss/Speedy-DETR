# Multiple Scales Fusion and Query Matching Stabilization for Detection with Transformer


## Model Overview:
![The architecture of Speedy-DETR. It consists of a backbone network, a multi-feature fusion module, an optional Transformer encoder, and a Transformer decoder. Multi-scale feature tokens undergo refinement through two encoder modules, SDE and HMoE, facilitated by a one-to-many matching strategy.](./model.png)



## Speedy DETR
- This repository is an official implementation of the paper Speedy-DETR: Multiple Scales Fusion and Query Matching Stabilization for Detection with Transformer.
- The code are built upon the official [RT-DETR](https://zhao-yian.github.io/RTDETR/) repository.
- The <font color=blue>**complete code**</font> will be released soon.
- **There is now a lot of duplicate, disorganised parameter code in the repository. <font color=blue>**Will be updated when we sort it out.**</font>**

### We publish the core algorithm, which can be embedded in several [DETR-like models](https://github.com/open-mmlab/mmdetection).
- **Note**: <font color=blue>All the core pseudocode, code and algorithmic details are in the paper, and we will subsequently publish a preprint version of the paper on [arXiv](https://arXiv.org).</font>

## Modules Detailed:
![Modules of Speedy-DETR. We first leverage features from the interleaved stages of the FPN, \(\{p_1, p_2, p_3, p_4\}\), as input to the encoder. The efficient modules then transform these multi-scale features into a sequence of image features using the Similarity-based Deduplication Encoder (SDE) and the Hybrid Multi-objects Encoder (HMoE). The SDE generates premium tokens post-deduplication, while the HMoE   module provides object-specific information, enhancing detail by generating target tokens. The OmPM is used to select positive samples, which serve as initial object queries for the decoder. Finally, the decoder, equipped with auxiliary prediction heads, iteratively optimizes these object queries to generate bounding boxes.](./module.png)


- Algorithm code in 
```
fu_detr:
    ----fusion/:
               ----**{fu__.py}**
````

## Quick Start

- prepare enviroment:

```bash
pip install -r requirements.txt
```





- Dataset:

- Download and extract COCO train and val images.
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

- Train:
```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/fudetr/fudetr_r50vd_6x_coco.yml -h 2 -a 0.3 -d 0.3
```

