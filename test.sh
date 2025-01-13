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