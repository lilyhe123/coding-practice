"""
question 21
Given an array of time intervals (start, end) for classroom lectures
(possibly overlapping), find the minimum number of rooms required.
For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.
-------------------

(0, 50), (30, 75), (60, 150)
First sort the intervals by the start time and then scan the intervals
from the first to the last.
we need to keep tracking a group of latest overlapped intervals. When
the next interval comes in, how to decide some intervals nees to remove
from the group and which one to remove? It's critical to this problem.
The answer is first-to-remove interval is the one with the minimum end time.
Compare the end time with the new start time, if the end time is no larger
than the start time then that interval need to be removed from the group.
Now we can see a heap is a proper data structure to hold the overlapped
intervals. It's a min_heap and the end time is to be compared in the heap.

min(end)>max(start)
"""

import heapq


def getMinNumberOfRooms(intervals):
    # sort the intervals by start time
    intervals.sort(key=lambda x: x[0])
    maxSize = 1
    # create a min_heap compared by the end time,
    # track the maximum size of the heap
    hq = []
    for start, end in intervals:
        while hq and hq[0] <= start:
            heapq.heappop(hq)
        heapq.heappush(hq, end)
        maxSize = max(maxSize, len(hq))
    return maxSize


def test_21():
    intervals = [(30, 50), (0, 20), (60, 150)]
    assert getMinNumberOfRooms(intervals) == 1
    intervals = [(30, 75), (0, 50), (60, 150)]
    assert getMinNumberOfRooms(intervals) == 2
    intervals = [(30, 75), (20, 40), (0, 50), (60, 150)]
    assert getMinNumberOfRooms(intervals) == 3
    intervals = [(30, 75), (20, 40), (0, 50), (35, 150)]
    assert getMinNumberOfRooms(intervals) == 4
