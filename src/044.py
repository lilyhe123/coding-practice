"""
question 44
We can determine how "out of order" an array A is by counting the number
of inversions it has.
Two elements A[i] and A[j] form an inversion if A[i] > A[j] but i < j.
That is, a smaller element appears after a larger element.
Given an array, count the number of inversions it has. Do this faster
than O(N^2) time.
You may assume each element in the array is distinct.

For example, a sorted list has zero inversions.
The array [2, 4, 1, 3, 5] has three inversions:
(2, 1), (4, 1), and (4, 3).
The array [5, 4, 3, 2, 1] has ten inversions: every distinct pair
forms an inversion.
-------------------

total pair count: n(n-1)/2=10

2, 4, 1, 3, 5
2, 4
^
1, 3, 5
^
merge sort
first half and second half

f(nums, start, end):

  mid =
  f(nums, start, mid)
  f(nums, mid+1, end)
  # merge
  i, j
  if nums[i] > nums[j]:
    count += (mid - i + 1)
"""


def inversionCount(nums):
    def mergeSort(nums, start, end, arr):
        if start == end or start > end:
            return 0

        # / true division, return a float; // floor division,
        # return an integer
        # !!! use // to return an integer
        mid = start + (end - start) // 2
        count = 0
        count += mergeSort(nums, start, mid, arr)
        count += mergeSort(nums, mid + 1, end, arr)
        # merge two subarray to a new array: [start, mid], [mid+1, end]
        i, j = start, mid + 1
        index = start
        while index <= end:
            if i > mid or (j <= end and nums[i] > nums[j]):
                arr[index] = nums[j]
                if i <= mid:
                    count += mid - i + 1
                j += 1
            else:
                arr[index] = nums[i]
                i += 1
            index += 1
        # copy the values in new arrays to nums
        for i in range(start, end):
            nums[i] = arr[i]
        return count

    arr = [0] * len(nums)
    return mergeSort(nums, 0, len(nums) - 1, arr)


def test_44():
    print(inversionCount([1, 2, 3, 4, 5]))
    print(inversionCount([2, 4, 1, 3, 5]))
    print(inversionCount([5, 4, 3, 2, 1]))
