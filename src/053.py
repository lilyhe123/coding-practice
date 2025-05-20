"""
question 53
Implement a queue using two stacks. Recall that a queue is a FIFO
(first-in, first-out) data structure with the following methods: enqueue,
which inserts an element into the queue, and dequeue, which removes it.
-------------------

put: s1.push(e)
get: if s2 not empty: s2.pop
     else: push all elements from s1 to s2, the s2.pop
s1: 4->5->6

s2: 3->2->1
first is the top in s2, last is the top in s1
"""


class Queue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def enqueue(self, e):
        self.s1.append(e)

    def dequeue(self):
        if not self.s2:
            if not self.s1:
                return None
            while self.s1:
                self.s2.append(self.s1.pop(-1))
        return self.s2.pop(-1)

    def size(self):
        return len(self.s1) + len(self.s2)


def test_53():
    queue = Queue()
    nums = [1, 2, 3, 4]
    for num in nums:
        queue.enqueue(num)
    print(queue.dequeue())

    nums = [5, 6, 7, 8]
    for num in nums:
        queue.enqueue(num)
    while queue.size() > 0:
        print(queue.dequeue())
