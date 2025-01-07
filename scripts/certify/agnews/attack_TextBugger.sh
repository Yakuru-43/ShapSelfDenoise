#!/bin/bash

# Sample script to run main.py with specific command-line arguments.
# You can modify the argument values as needed.

python main.py \
    --mode attack \
    --method TextBugger \
    --config config.yml \
    --batchsize 16 \
    --sample-size 100 \
    --precision full \
    --dataset agnews \
    --mask_word "<mask>"