#!/bin/bash

# Sample script to run main.py with specific command-line arguments.
# You can modify the argument values as needed.
CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --mode evaluate_lime \
    --method DeepWordBug \
    --defence lime \
    --n 100\
    --config config.yml \
    --batchsize 16 \
    --sample-size 100 \
    --precision half \
    --dataset agnews \
    --mask_word "<mask>"
