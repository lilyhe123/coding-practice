"""
question 34
Given a string, find the palindrome that can be made by inserting the fewest
number of characters as possible anywhere in the word. If there is more than
one palindrome of minimum length that can be made, return the
lexicographically earliest one (the first one alphabetically).

For example, given the string "race", you should return "ecarace", since we
can add three letters to it (which is the smallest amount to make a palindrome)
. There are seven other palindromes that can be made from "race" by adding
three letters, but "ecarace" comes first alphabetically.

As another example, given the string "google", you should return "elgoogle".
-------------------

1. understand the problem
what's the input and output?
input is a string, output is s palindrome string after insert some
character to the input string.

what's the constraints?
can only inserting new chars to make output string palindrome
if multiple palindromes with the same length exists, return the
lexicographically earliest one.

2. brainstorm
first we need to find the palindrome subsequence with max length
top-down
f(arr, i, j):
  if i == j: return 1
  if i > j: return 0
  if arr[i] == arr[j]:
    return 2 + f(arr, i+1, j-1)
  else:
    return max(f(arr, i+1, j), f(arr, i, j-1))

bottom-up
start|end
  l e e o e
l 1 1 2 2 3
e   1 2 2 3
e.    1 1 3
o.      1 1
e.        1

leetcode
  ^.   ^
Output: 5
Explanation: Inserting 5 characters the string becomes "leetcodocteel".
"""


def minInsertions(s: str) -> int:
    n = len(s)
    dp = [[0 for _ in range(n)] for _ in range(n)]
    for step in range(n):
        i = 0
        while i + step < n:
            j = i + step
            if step == 0:
                dp[i][j] = 1
            elif step == 1:
                if s[i] == s[j]:
                    dp[i][j] = 2
                else:
                    dp[i][j] = 1
            else:
                if s[i] == s[j]:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
            i += 1

    return n - dp[0][n - 1]


def test_34():
    print(minInsertions("leetcode"))
