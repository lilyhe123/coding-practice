# uncomments the following line when need to debug stack overflow error
# import sys
# sys.setrecursionlimit(10)

# 1 - 50
"""
!! question 1
Given a list of numbers and a number k, return whether any two numbers from
the list add up to k.
For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
Bonus: Can you do this in one pass?
-------------------

## Understand the problem
The input is a list of numbers and a number k, the output is a boolean...

## !! brain storming
1. brute-force approach
the most straight-forward approach is to find every possible pair of numbers
 in the list,
For every pair check whether the sum is equal to k
time O(n^2), space O(1)

2. sort the array and use two-pointers to find the pair that sum up to k.
3, 7, 10, 15
   ^
      ^
18 > 17, 13 < 17, 17 == 17 return true
Time O(nlgn), space O(1)

3. !!  to do this in one pass? This means we can only iterate the list once.
 We need an effective way to check whether we've already encountered the
 complement of the current number needed to reach K.
So we can introduce a hashset data structure to add all the already-visited
 numvers to it.
10, 15, 3, 7
7   2   14 10       ^
k=17
set: [10, 15, 3] return true
"""


def test_1():
    pass
