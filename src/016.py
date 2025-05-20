"""
question 16
You run an e-commerce website and want to record the last N order ids in a
log. Implement a data structure to accomplish this, with the following API:

record(order_id): adds the order_id to the log
get_last(i): gets the ith last element from the log.
i is guaranteed to be smaller than or equal to N.
You should be as efficient with time and space as possible.
-------------------

N=3
record: 1,2,3,4,5,6,5
[5,6,4]
record(i):
  if i in the list, move i to the head of the queue
  if i not in the list, add i to the head of the queue.
  If the len(queue) > N, remove the last element

list: []
"""

from collections import deque


class OrderTracker:
    def __init__(self, N):
        self.N = N
        # id -> idx in queue
        self.queue = deque()

    def record(self, id):
        self.queue.append(id)
        if len(self.queue) > self.N:
            self.queue.popleft()

    def get_last(self, i):
        return self.queue[i]


def test_16():
    log = OrderTracker(3)
    ids = [1, 2, 3, 4, 5, 6]
    for id in ids:
        log.record(id)
    print(log.get_last(1))
