#!/bin/bash

# Run the mask-rate accuracy sweep (the curve in Figure 2) on SST-2.
# certify mode loops the mask rate from 0.1 to 0.9 and, for each rate,
# SHAP-masks the top words, denoises, classifies and logs the accuracy.
# It reads dataset/sst2/dataset_certify.json.

python main.py \
    --mode certify \
    --config config.yml \
    --batchsize 16 \
    --sample-size 100 \
    --precision full \
    --dataset sst2 \
    --mask_word "<mask>"
