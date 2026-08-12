#!/bin/sh

# getting last commit message "https://stackoverflow.com/questions/7293008/display-last-git-commit-comment"
message=$(git log -2 --pretty=%B)

if [[ "$message" == *$'\n\n'* ]]; then
  echo "Subject is separated from body by a blank line."
else
  echo "Subject is not separated from body by a blank line!"
  exit 1
fi

# message rules "https://www.gitkraken.com/learn/git/best-practices/git-commit-message"
subject=$(echo "$message" | head -n 1)
body=$(echo "$message" | tail -n +3)

word_list=("Added:" "Updated:" "Removed:" "Bugfix:")

# extract the keyword from the subject
keyword=$(echo "$subject" | awk '{print $1}')

# function to check if a keyword is in the list
found=false
for item in "${word_list[@]}"; do
  if [[ "$item" == "$keyword" ]]; then
    echo "Keyword {$item} found."
    found=true
    break
  fi
done

# if the keyword was found
if ! $found; then
  echo "Missing change keyword: {$keyword}!"
  exit 1  # not found
fi

# check that subject line is within 50 characters
if [[ ${#subject} -le 50 ]]; then
  echo "Subject line is within 50 characters."
else
  echo "Subject line is greater than 50 characters!"
  exit 1
fi

# check for capital letter
if [[ "$subject" =~ ^[A-Z] ]]; then
  echo "Subject line starts with a capital letter."
else
  echo "Subject line does not start with a capital letter!"
  exit 1
fi

# no period at end
if [[ ! "$subject" =~ \.$ ]]; then
  echo "Subject line does not end with a period."
else
  echo "Subject line should not end with period!"
  exit 1
fi

# check for body
if [[ -n "$body" ]]; then
  echo "Commit body exits."
else
  echo "Commit body is missing!"
  exit 1
fi

# check body length
if echo "$body" | grep -qE ".{101,}"; then
  echo "Body exceeds 100 characters!"
  exit 1
else
  echo "Body within 100 characters."
fi
