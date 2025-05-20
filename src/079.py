"""
question 79
Given an array of integers, write a function to determine whether the array
could become non-decreasing by modifying at most 1 element.

For example, given the array [10, 5, 7], you should return true, since we can
modify the 10 into a 1 to make the array non-decreasing.

Given the array [10, 5, 1], you should return false, since we can't modify any
one element to get a non-decreasing array.

-------------------
Scan the array from left to right, there should be at most one decreaing happen.
If the decreasing happend at the second element, we can change the first element
to a value less than the second one. So return true.
If the decreasing happend at the last element, we can change the last element to
a value larger than its previous one. So return true.
But if the descreasing happens in the middle, it depends.
There is a valley we need to check the previous element and next element of the valley.
if pre <= next, we can change the valley to a vlue in between. otherwise we can not.

"""


def canBecomeNondecreasing(nums):
    count = 0
    valley_idx = 0
    for i in range(1, len(nums)):
        if nums[i] < nums[i - 1]:
            count += 1
            valley_idx = i
        if count > 1:
            return False
    # count = 0 or 1
    return (
        count == 0
        or valley_idx == 1
        or valley_idx == len(nums) - 1
        or nums[valley_idx - 1] <= nums[valley_idx + 1]
    )


def test_79():
    nums = [10, 5, 7]
    assert canBecomeNondecreasing(nums)
    nums = [10, 5, 1]
    assert not canBecomeNondecreasing(nums)
    nums = [1, 5, 2, 7]
    assert canBecomeNondecreasing(nums)
    nums = [1, 5, 2, 3]
    assert not canBecomeNondecreasing(nums)
    nums = [1, 2, 3, -10]
    assert canBecomeNondecreasing(nums)


test_79()
