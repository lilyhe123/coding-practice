"""
question 75
Given an array of numbers, find the length of the longest increasing subsequence
in the array. The subsequence does not necessarily have to be contiguous.

For example, given the array
[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15],
the longest increasing subsequence has length 6: it is 0, 2, 6, 9, 11, 15.

-------------------
[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
 1  2  2   3  2   3  3   4  2  4  3   4  3   5  4   6

bottom-up dynamic programming
create dp array with length N
For i-th element in dp array, it represent the size of longest subsequence ending
in ith element in the given array.

For i-th element in dp array
  dp[i] = max(dp[j]+1) when nums[i] > nums[j],  j in range [0, i-1]

time O(N^2), space O(N), N is the length of given array
"""


def getLIS(nums):
    N = len(nums)
    dp = [0] * N
    dp[0] = 1
    longest = 1
    for i in range(1, N):
        for j in range(0, i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
        longest = max(dp[i], longest)
    return longest


def test_75():
    nums = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    assert getLIS(nums) == 6
    nums = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15, 8, 7, 3]
    assert getLIS(nums) == 6


test_75()
