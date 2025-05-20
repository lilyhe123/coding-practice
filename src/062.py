"""
question 62
There is an N by M matrix of zeroes. Given N and M, write a function to count
the number of ways of starting at the top-left corner and getting to the
bottom-right corner. You can only move right or down.

For example, given a 2 by 2 matrix, you should return 2, since there are two
ways to get to the bottom-right:

Right, then down
Down, then right
Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.
-------------------

f(n, m) = f(n, m-1) + f(n-1, m)
1 1 1  1  1
1 2 3  4  5
1 3 6  10 15
1 4 10 20 35
1 5 15 35 70

time O(n*m), space O(m)
"""


def totalWaysToMove(n, m):
    if n == 1:
        return 1
    pre = [1] * m
    cur = [0] * m
    for i in range(2, n + 1):
        cur[0] = 1
        for j in range(1, m):
            cur[j] = cur[j - 1] + pre[j]
        pre = cur
        cur = [0] * m
    return pre[m - 1]


def test_62():
    print("run tests62")
    assert totalWaysToMove(2, 2) == 2
    assert totalWaysToMove(5, 5) == 70
