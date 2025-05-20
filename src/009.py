"""
question 9
This problem was asked by Airbnb.

Given a list of integers, write a function that returns the largest sum of
non-adjacent numbers. Numbers can be 0 or negative.

For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5.
[5, 1, 1, 5] should return 10, since we pick 5 and 5.

Follow-up: Can you do this in O(N) time and constant space?
-------------------

Thinking
[2, 4, 6, 2, 5]
 ^
 contraints:
  - sum of non-adjacent numbers
 1. brute-force approach
iterate the array from the first to the last. For every element,
mostly there are two options: select it or not. If has chose the previous
element, then there is only one option to the current, not select.
time O(2^n), space O(1)

2. DP
dp[i]: the max sum of non-adjacent numbers ending in the ith element
2, 4, 6, 2, 5
2. 4, 8, 6, 13
dp[i] = max(dp[i-1], dp[i-2], dp[i-2]+nums[i])
"""


def maxSum4NonAdjacent(nums: list[int]) -> int:
    dp0, dp1, dp2 = nums[0], max(nums[0], nums[1]), 0
    for i in range(2, len(nums)):
        dp2 = max(max(dp0, dp1), dp0 + nums[i])
        dp0 = dp1
        dp1 = dp2
    return dp1


def test_9():
    assert maxSum4NonAdjacent([2, 4, 6, 2, 5]) == 13
    assert maxSum4NonAdjacent([10, -1, 1, 2]) == 12


test_9()
