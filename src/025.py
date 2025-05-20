"""
question 25
Implement regular expression matching with the following special characters:

. (period) which matches any single character
* (asterisk) which matches zero or more of the preceding element
That is, implement a function that takes in a string and a valid regular
expression and returns whether or not the string matches the regular expression.

For example, given the regular expression "ra." and the string "ray", your
function should return true. The same regular expression on the string
"raymond" should return false.

Given the regular expression ".*at" and the string "chat", your function should
return true. The same regular expression on the string "chats" should return false.
-------------------

1. recursion and backtracking
input: pattern, text
f(i1, i2):
for ., i1++, i2++
for *, i1++, i2: i2, i2+1,...i2+n
others: if pattern[i1] == text[i2]: i1++, i2++; otherwise return

2. In solution #1 we might calculate the same f(i1, i2) multiple times.
To reduce the duplication, use DP.

populate the DP matrix row by row, only process the up-right half
  c h s a t s a
c y n n n n n n
.   y y y y y y
*     y y y y y
t       n y n n
.         n y y

for any cell, except (0,0), two possible directions to reach here:
1. from up-left
2. from left, only when current char in the pattern is '.'.

time O(N*M), space O(M)
"""


def regexMatched(pattern: str, text: str) -> bool:
    N, M = len(pattern), len(text)
    if N > M:
        return False
    pre = [0] * M
    cur = [0] * M
    for row in range(0, N):
        p = pattern[row]
        for col in range(row, M):
            t = text[col]
            # reach from up-left
            if (col == 0 or pre[col - 1]) and (p == "*" or p == t or p == "."):
                cur[col] = 1
            # reach from left
            elif (col == 0 or cur[col - 1]) and p == ".":
                cur[col] = 1
        # no valid path in this row, no chance to match, fail early
        if sum(cur) == 0:
            return False
        pre = cur
        cur = [0] * M
    return pre[M - 1] == 1


def test_25():
    assert regexMatched("c.*t.", "chsatsa")
    assert regexMatched(".*at", "chat")
    assert regexMatched(".*at", "sdfdedat")
    assert not regexMatched("chi.", "chat")
    assert regexMatched("ch*.", "chatsesd")
    assert regexMatched("*.", "blablabla")
    assert regexMatched(".a*", "blablablaay")
    assert not regexMatched("*a.", "blablablaay")
