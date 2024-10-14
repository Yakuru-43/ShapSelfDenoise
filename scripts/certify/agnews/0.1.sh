#!/bin/bash

# Sample script to run main.py with specific command-line arguments.
# You can modify the argument values as needed.

python main.py \
    --mode certify \
    --config config.yml \
    --batchsize 3 \
    --sample-size 100 \
    --precision full \
    --dataset agnews \
    --maskrate 0.1 \
    --mask_word "<mask>"