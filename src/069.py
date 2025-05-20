"""
question 69
Given a list of integers, return the largest product that can be made by
multiplying any three integers.

For example, if the list is [-10, -10, 5, 2], we should return 500, since that's
-10 * -10 * 5.

You can assume the list has at least three integers.

-------------------
If all positive integers, the largest product is contribute by the largest 3.
But negative ones can be included, product of two negative ones might contribute
to the largest product.
So let's do this.
get largest 3(l1, l2, l3) and smallest 2 (s1,s2).
if l1 <= 0, (all negative), rst = l1 * l2 * l3
elif s1 <= 0, s2 <= 0,  rst = l1 * max(l2*l3, s1*s2)

Get largest 3, use a min_heap with size 3. time O(nlg3) -> O(n)
Get smallest 2, use a max_heap with size 2. time O(nlg2) -> O(n)

time O(n), space O(1)
"""

import heapq


def getLargestProducct(nums):
    min_size, max_size = 3, 2
    minq, maxq = [], []
    for num in nums:
        heapq.heappush(minq, num)
        heapq.heappush(maxq, -num)
        if len(minq) > min_size:
            heapq.heappop(minq)
        if len(maxq) > max_size:
            heapq.heappop(maxq)

    # largest 3 in minq
    p1 = heapq.heappop(minq) * heapq.heappop(minq)
    largest = minq[0]
    if largest <= 0:
        return largest * p1
    else:
        # smallest 2 in maxq
        p2 = maxq[0] * maxq[1]
        return largest * max(p1, p2)


def test_69():
    nums = [-10, -10, 5, 2]
    assert getLargestProducct(nums) == 500
    nums = [-10, 3, 5, 2]
    assert getLargestProducct(nums) == 30
    nums = [-10, -3, 3, 5, 2]
    assert getLargestProducct(nums) == 150
    nums = [-2, -3, 3, 10, 3]
    assert getLargestProducct(nums) == 90
    nums = [-2, -3, 10]
    assert getLargestProducct(nums) == 60
    nums = [-2, -3, -10, -5, -1]
    assert getLargestProducct(nums) == -6
    nums = [-2, -3, -10, -5, 0]
    assert getLargestProducct(nums) == 0
