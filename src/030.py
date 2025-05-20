"""
question 30
You are given an array of non-negative integers that represents a
two-dimensional elevation map where each element is unit-width wall and the
integer is the height. Suppose it will rain and all spots between two walls get
filled up.

Compute how many units of water remain trapped on the map in O(N) time and O(1)
space.

For example, given the input [2, 1, 2], we can hold 1 unit of water in the
middle.

Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2 in
the second, and 3 in the fourth index (we cannot hold 5 since it would run off
to the left), so we can trap 8 units of water.

-------------------
[2,1,2]
|   |
| | |
for ith element,
- left_max_wall is the max height in heights[0, i],
- right_max_wall is the max height in heights[i, N-1]
so the water height is min(left_max_wall, right_max_wall) - cur_wall_height

[2,1,2]
left_max_array: [2,2,2]
right_max_array:[2,2,2]


left_max_array = [0] * N
left_max_array[0] = heights[0]
for i in range(1, len(heights)):
    left_max_array[i] = max(left_max_array[i-1], heights[i])
right_max_array = [0] * N
right_max_array[N-1] = heights[N-1]
for i in range(N-2, -1, -1):
    right_max_array = max(right_max_array(i+1), heights[i])
time O(N), space O(N)
TODO how to optimize to time O(N), space O(1)
"""


def totalTrappedWater(heights):
    N = len(heights)
    left_max_array = [0] * N
    left_max_array[0] = heights[0]
    for i in range(1, N):
        left_max_array[i] = max(left_max_array[i - 1], heights[i])
    right_max_array = [0] * N
    right_max_array[N - 1] = heights[N - 1]
    for i in range(N - 2, -1, -1):
        right_max_array[i] = max(right_max_array[i + 1], heights[i])
    water = 0
    for i in range(N):
        water += min(left_max_array[i], right_max_array[i]) - heights[i]
    return water


def test_30():
    assert totalTrappedWater([2, 1, 2]) == 1
    assert totalTrappedWater([3, 0, 1, 3, 0, 5]) == 8
