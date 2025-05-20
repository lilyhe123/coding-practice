"""
question 60
Given a multiset of integers, return whether it can be partitioned into two
subsets whose sums are the same.

For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return
true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35},
which both add up to 55.

Given the multiset {15, 5, 20, 10, 35}, it would return false, since we
can't split it up into two subsets that add up to the same sum.
-------------------

target = 55
{15, 5, 20, 10, 35, 15, 10}
 ^
   0 1 2 3 4 5 6 7 8 9 10.   15.   20   55
0  y n n
15 y                          y
5. y         y                y.   y
20
change to problem to find a subset to a target value

1. brute-force: for every number, two options: choose it or not. Iterate all
combination and calculate the sum, if the sum is equal to the target, return
True. time O(2^n), space O(1).

2. recursion with memorization
  !! how to analyze the time and space complexity
  time O(n*m), space O(n*m), n is the size of given array and m is the half
  of the sum of the given array.

"""


def canPartition(nums):
    total = sum(nums)
    if total % 2 == 1:
        return False
    target = total // 2
    memo = set()

    def dfs(i, remaining):
        if remaining == 0:
            return True
        if target < 0 or i == len(nums) or (i, remaining) in memo:
            return False

        rst = dfs(i + 1, remaining) or dfs(i + 1, remaining - nums[i])
        if not rst:
            memo.add((i, remaining))
        return rst

    return dfs(0, target)


def test_60():
    nums = [15, 5, 20, 10, 35, 15, 10]
    assert canPartition(nums) is True
    nums = [15, 5, 20, 10, 35]
    assert canPartition(nums) is False
