"""
question 46
Given a string, find the longest palindromic contiguous substring.
If there are more than one with the maximum length, return any one.
For example, the longest palindromic substring of "aabcdcb" is "bcdcb".
The longest palindromic substring of "bananas" is "anana".
-------------------

1. brute-force approach
check every possible substring from length n to length 2.
total number of substring is n^2. To check whether a substring is
palindromic, take linear time.
time O(n^3), space O(1).

2. optimize
bottom-up DP: time O(n^2), space O(n^2)
f(str, start, end):
   if start == end: return True
   if start == end - 1: return str[start] == str[end]

  if str[start] == str[end] && f(str, start+1, end-1): return True
  if str[start] != str[end]: return False

len(substring): 1 to n
bananas
start|end b a n a n a s
       b  y n n
       a.   y n y
       n      y.n y
       a.       y n y
       n.         y n
       a.           y n
       s              y
"""


def getLongestPalindrome(s):
    def isPalindrome(s, start, end, memo):
        if start >= end:
            return True
        key = (start, end)
        if key not in memo:
            if s[start] == s[end] and isPalindrome(s, start + 1, end - 1, memo):
                memo[key] = 1
            else:
                memo[key] = 0

        return memo[key] == 1

    memo = {}  # (i,j) -> 0|1 false|true
    for length in range(len(s) - 1, 0, -1):
        i = 0
        while i + length < len(s):
            if isPalindrome(s, i, i + length, memo):
                return s[i : i + length + 1]
            i += 1
    print(memo)
    return s[0:1]


def test_46():
    print(getLongestPalindrome("bananas"))
