This code is based on the official training recipe for ImageNet classification provided by [Torchvision](https://github.com/pytorch/vision/tree/main/references/classification).

# Image classification reference training scripts

This folder contains reference training scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

### SwinTransformer
```
torchrun --nproc_per_node=8 train.py\ 
--model $MODEL --epochs 300 --batch-size 128 --opt adoptw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224
```
Here `$MODEL` is one of `swin_t`, `swin_s` or `swin_b`.
Note that `--val-resize-size` was optimized in a post-training step, see their `Weights` entry for the exact value.
