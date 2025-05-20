"""
question 4
Given an array of integers, find the first missing positive integer in linear
time and constant space. In other words, find the lowest positive integer
that does not exist in the array. The array can contain duplicates
and negative numbers as well.

For example, the input [3, 4, -1, 1] should give 2.
The input [1, 2, 0] should give 3.

You can modify the input array in-place.
-------------------

 1  2   3  4
 2 3 1 4
 1 2 3 4
[3, 4, -1, 1]
[1, -1, 3, 4]
for i in range():
  val = nums[i]
  if val in range(1, n+1):
    # swap elements in i and val+1
nums[i] = i+1
"""


def findLowest(nums: int) -> int:
    n = len(nums)
    for i in range(n):
        while nums[i] in range(1, n + 1) and i + 1 != nums[i]:
            # swap element in i and nums[i]-1
            j = nums[i] - 1
            tt = nums[i]
            nums[i] = nums[j]
            nums[j] = tt
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


def test_4():
    assert findLowest([3, 4, -1, 1]) == 2
    assert findLowest([1, 2, 0]) == 3
