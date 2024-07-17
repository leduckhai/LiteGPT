# LiteGPT
```
Le Duc Khai
University of Toronto
```

## Dataset structure
```
root
├── images
│   ├── train
│   └── test
├── annotations
│   ├── train
│   │   └── grounded_diseases_train.json
│   └── test
│       └── grounded_diseases_test.json
└── pretrained_checkpoint
    └── checkpoint_stage3.pth
```

You may load from the pretrained model checkpoints:

For `checkpoint_stage3.pth`, you can load from the pretrained model below:
| MiniGPT-v2 (after stage-3) |
|------------------------------|
|[Download](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing) |

## Installation
- Python == 3.10.13
```bash
git clone https://github.com/nngocson2002/LVLM-Med.git
cd LVLM-Med
pip install -r requirements.txt
```

## Training
### Set the visual encoder
We provide different visual encoders with the following keywords:
 - `eva_clip_g`
 - `pubmed_clip_vit`
 - `biomed_clip`
 - `biomed_pubmed_clip`

After selecting the visual encoder you want, set it [here](train_configs/train_vindrcxr.yaml#L7) at Line 7, and [here](eval_configs/eval_biomedclip_llama.yaml#L8) at Line 8.

### Set Paths for Training
- Set the training image path to `root/images/train` [here](medlvlm/configs/datasets/vindrcxr/default.yaml#L5) at Line 5.
- Set the training annotations path to `root/annotations/test/grounded_diseases_train.json` [here](medlvlm/configs/datasets/vindrcxr/default.yaml#L6) at Line 6.
- Set the pretrained checkpoint path to `root/pretrained_checkpoint/checkpoint_stage3.pth` [here](train_configs/train_vindrcxr.yaml#L9) at Line 9.
- Set the checkpoint save path [here](train_configs/train_vindrcxr.yaml#L44) at Line 44.
- If you set `wandb_log` to `true` [here](train_configs/train_vindrcxr.yaml#L57) at Line 57, you need to set the `wandb_token` [here](train_configs/train_vindrcxr.yaml#L58) at Line 58 to enable tracking.

### Set Paths for Evaluation (After Training)
- Set the evaluation annotations path to `root/annotations/test/grounded_diseases_test.json` [here](eval_configs/eval_vindrcxr.yaml#L27) at Line 27.
- Set the evaluation image path to `root/images/test` [here](eval_configs/eval_vindrcxr.yaml#L28) at Line 28.
- Set the evaluation result output path [here](eval_configs/eval_vindrcxr.yaml#L38) at Line 38.
- Set the prompt you want to evaluate the model with [here](eval_configs/eval_vindrcxr.yaml#L29) at Line 29.

### Run
```bash
torchrun --nproc-per-node NUM_GPU train.py\ 
         --cfg-path train_configs/train_vindrcxr.yaml\
         --cfg-eval-path eval_configs/eval_vindrcxr.yaml\
         --eval-dataset vindrcxr_val
```

## Evaluation
If you want to evaluate the model independently instead of during training, follow the [step 2](#set-paths-for-evaluation-after-training) in the Training section, and then run:
```bash
torchrun --nproc-per-node NUM_GPU evaluate.py\ 
         --cfg-path eval_configs/eval_vindrcxr.yaml\
         --eval-dataset vindrcxr_val
```