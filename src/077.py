"""
question 77
Given a list of possibly overlapping intervals, return a new list of intervals
where all overlapping intervals have been merged.

The input list is not necessarily ordered in any way.

For example, given [(1, 3), (5, 8), (4, 10), (20, 25)], you should return [(1,
3), (4, 10), (20, 25)].

-------------------
1. Sort the interval list by start time.
2. use two pointers to iterate the sorted list.
   p1 points to the last merged interval, p2 points to the interval to check.
   if the two intervals are overlapped:
     merge them and save the merged interval to first interval. p2 moves forward.
   othewise,
     p1 moves forward and save the second list to p1. p2 moves forward.
time O(n), space O(1)
"""


def mergeIntervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    p1 = 0
    for p2 in range(1, len(intervals)):
        s1, e1 = merged[p1]
        s2, e2 = intervals[p2]
        if s2 <= e1:  # overlapped
            merged[p1] = (s1, max(e1, e2))
        else:
            p1 += 1
            merged.append(intervals[p2])
    return merged


def test_77():
    intervals = [(1, 3), (5, 8), (4, 10), (20, 25)]
    assert mergeIntervals(intervals) == [(1, 3), (4, 10), (20, 25)]
    intervals = [(1, 10), (15, 21), (3, 8), (20, 25), [11, 18], [30, 50]]
    assert mergeIntervals(intervals) == [(1, 10), (11, 25), [30, 50]]


if __name__ == "__main__":
    test_77()
