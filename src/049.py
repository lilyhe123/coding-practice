"""
question 49
Given an array of numbers, find the maximum sum of any contiguous
subarray of the array.

For example, given the array [34, -50, 42, 14, -5, 86],
the maximum sum would be 137,
since we would take elements 42, 14, -5, and 86.

Given the array [-5, -1, -8, -9], the maximum sum would be 0,
since we would not take any elements.

Do this in O(N) time.
-------------------

[34, -50, 42, 14, -5, 86]
      ^
"""


def maxSum(nums):
    sum = 0
    maxSum = 0
    for num in nums:
        sum += num
        if sum <= 0:
            sum = 0
        maxSum = max(maxSum, sum)
    return maxSum


def test_49():
    print(maxSum([34, -50, 42, 14, -5, 86]))
    print(maxSum([-5, -1, -8, -9]))
