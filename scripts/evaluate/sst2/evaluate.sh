#!/bin/bash

# Statistical evaluation of the (non-deterministic) LIME defence on SST-2.
# Requires the NoDefence attack output to exist first, i.e. run
#   scripts/attack/sst2/attack_DeepWordBug.sh
# which produces out/attack/sst2/DeepWordBug/NoDefence/half/dataset.csv

CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --mode evaluate_lime \
    --method DeepWordBug \
    --defence lime \
    --n 100 \
    --config config.yml \
    --batchsize 16 \
    --sample-size 100 \
    --precision half \
    --dataset sst2 \
    --mask_word "<mask>"
