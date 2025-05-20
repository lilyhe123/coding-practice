"""
question 17
Suppose we represent our file system by a string in the following manner:
The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:

dir
    subdir1
    subdir2
        file.ext
The directory dir contains an empty sub-directory subdir1 and a sub-directory
subdir2 containing a file file.ext.

The string
"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" # noqa: E501
represents:
dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
The directory dir contains two sub-directories subdir1 and subdir2.
subdir1 contains a file file1.ext and an empty second-level sub-directory
subsubdir1. subdir2 contains a second-level sub-directory subsubdir2 containing
a file file2.ext.

We are interested in finding the longest (number of characters) absolute path
to a file within our file system. For example, in the second example above,
the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its length
is 32 (not including the double quotes).

Given a string representing the file system in the above format,
return the length of the longest absolute path to a file in the abstracted
file system. If there is no file in the system, return 0.
-------------------

longestLen = 0
tokens = str.split('/n')
paths = []
for token in tokens:
  if not paths: paths.append(token)
  else:
    count = token.sub('/t') # todo: how to count the substring
    token remove all /t # todo: how to remove all substrings
    paths = paths[:count]
    if isFile(token):
      size = len(token)
      for path in paths:
        size += len(path)
      size += len(paths)
      longestLen = max(longestLen, size)
    else:
      paths.append(token)
time O(n), space O(n), n is the token count in the given string
"""


def getLongestFilePath(s):
    longestPath = 0
    token_list = s.split("\n")
    path_list = []
    for token in token_list:
        count = token.count("\t")  # !! count the substring
        token = token.replace("\t", "")  # !! remove or replace all substrings
        path_list = path_list[:count]
        if token.endswith(".ext"):
            size = len(token)
            for path in path_list:
                size += len(path)
            size += len(path_list)
            longestPath = max(longestPath, size)
        else:
            path_list.append(token)
    return longestPath


def test_17():
    print("run test17")
    s = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
    assert getLongestFilePath(s) == 20
    s = "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"  # noqa: E501
    assert getLongestFilePath(s) == 32
