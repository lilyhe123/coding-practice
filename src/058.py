"""
question 58
An sorted array of integers was rotated an unknown number of times.

Given such an array, find the index of the element in the array in faster
than linear time. If the element doesn't exist in the array, return null.

For example, given the array [13, 18, 25, 2, 8, 10] and the element 8,
return 4 (the index of 8 in the array).

You can assume all the integers in the array are unique.
-------------------

Try to use binary search to find the target value.

There is a turning point in the array.
for any subrange[left, right]
- if left > right, the turning point is included. The subarray is not sorted
- if left < right, the turning point is not included. The subarray is sorted.

So we can only rely on the sorted subarray to check whether the target
is in it or not. If yes, we keep search in this half,
otherwise we search on the other halp

start, mid, end
if target == nums[mid]: return mid
if nums[start] < nums[mid]:
  if nums[start] <= target < nums[mid]: end = mid - 1
  else: start = mid + 1
elif nums[mid] < nums[end]:
  if nums[mid] < target <= nums[end]: start = mid + 1
  else: end = mid - 1
"""


def findIndex(nums, target):
    def binarySearch(start, end):
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            if nums[start] <= nums[mid]:  # this is the sorted half
                if nums[start] <= target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
        return None

    return binarySearch(0, len(nums) - 1)


def test_58():
    nums = [13, 18, 25, 2, 8, 10]
    target = 8
    print(findIndex(nums, target))  # Output: 4
    target = 16
    print(findIndex(nums, target))  # Output: None
