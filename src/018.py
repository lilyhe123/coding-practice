"""
!! question 18
Given an array of integers and a number k, where 1 <= k <= length of the array,
 compute the maximum values of each subarray of length k.

For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get:
[10, 7, 8, 8], since:

10 = max(10, 5, 2)
7 = max(5, 2, 7)
8 = max(2, 7, 8)
8 = max(7, 8, 7)
Do this in O(n) time and O(k) space. You can modify the input array in-place
and you do not need to store the results. You can simply print them out
as you compute them.
-------------------

10, 5, 2, 7, 8, 7
   |      |
1. brute-force approach
time O((nk), space O(1)

2. use heap to get the max one with constant time
  max_heap with size of k,
  init: time O(klogk),
  window moving forward: remove the first one and add a new one, time O(k)
  time O(kn), space O(k)

3. deque to store indexes of elements in the current window but remove those
   useless (no chance to be the max) elements.
   If a larger element comes in, remove all the smaller elements in the queue.
   So elements storing in the queue are always in decreasing order.
    The first one in the queue is always the max number in the current window.
    time O(n), space O(k)
10, 5, 2, 7, 8, 7
"""

from collections import deque


def getMaxInSubarray(nums, k):
    # the queue is to store indexes of elements in the currrent windoq
    queue = deque()
    output = []
    # initialize the queue with the first k numbers
    for i in range(k):
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop()
        queue.append(i)
    output.append(nums[queue[0]])
    # side the window across the array
    for start in range(1, len(nums) - k + 1):
        end = start + k - 1
        # remove indexes that are out of the window
        # actually since we move one space forward,
        # mostly we remove only one element
        if queue and queue[0] < start:
            queue.popleft()
        # add a new number to the end
        while queue and nums[queue[-1]] < nums[end]:
            queue.pop()
        queue.append(end)
        # first one in the queue is always the max number in the current window
        output.append(nums[queue[0]])
    return output


def test_18():
    print("run test18")
    nums = [10, 5, 2, 7, 8, 7]
    k = 3
    assert getMaxInSubarray(nums, k) == [10, 7, 8, 8]
    # increasing array
    nums = [1, 5, 10, 12, 18, 20, 21, 25, 29, 30]
    k = 5
    assert getMaxInSubarray(nums, k) == [18, 20, 21, 25, 29, 30]
    # decreasing array
    nums = [30, 29, 25, 21, 20, 18, 12, 10, 5, 1]
    k = 5
    assert getMaxInSubarray(nums, k) == [30, 29, 25, 21, 20, 18]
