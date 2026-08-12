Key of Understanding Pipeline Checks
Rules:

As outlined in this website [https://www.gitkraken.com/learn/git/best-practices/git-commit-message] a good commit message has a subject and a body.
The Subject should start with a keyword and be at most 50 characters long.


The body should be at most 100 characters long and does not end with a period.

Running

The chceck commits file should run alongside the pipeline so running it locally would not be necessary. However, if need be:
- Save the file
- Run the command: chmod u+x. This will allow for execution of the file so it can be tested / ran.
- Then type ./check-commits.sh in the terminal and this will run the file.

Keywords

These are the keywords that must be included in the subject of the commit message.
- Added: -> This is for adding features or files to the directory.
- Removed: -> This is for removing features or files from the directory.
- Updated: -> This is when updating fetures or files in the directory.
- Security: -> Use when updating security features in the application.

IMPORTANT!

Make sure that the ':' is part of the keyword!