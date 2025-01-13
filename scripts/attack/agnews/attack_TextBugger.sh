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

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then

  # Stage all changes
  git add -A

  # Commit changes with a timestamp
  commit_message="Automated commit on $(date '+%Y-%m-%d %H:%M:%S')"
  git commit -m "$commit_message"

  # Push to the master branch
  git push origin master

  echo "Changes pushed to GitHub successfully."

else
  echo "No changes to commit."
fi