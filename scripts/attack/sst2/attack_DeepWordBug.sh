#!/bin/bash

# Sample script to run main.py with specific command-line arguments.
# You can modify the argument values as needed.
# CUDA_VISIBLE_DEVICES=1 \
python main.py \
    --mode attack \
    --method DeepWordBug \
    --config config.yml \
    --batchsize 16 \
    --sample-size 100 \
    --precision half \
    --dataset sst2 \
    --mask_word "<mask>"

# # Check for uncommitted changes
# if ! git diff-index --quiet HEAD --; then

#   # Stage all changes
#   git add -A

#   # Commit changes with a timestamp
#   commit_message="Automated commit on $(date '+%Y-%m-%d %H:%M:%S')"
#   git commit -m "$commit_message"

#   # Push to the master branch
#   git push origin master

#   echo "Changes pushed to GitHub successfully."

# else
#   echo "No changes to commit."
# fi