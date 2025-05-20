"""
question 78
Given k sorted singly linked lists, write a function to merge all the lists into
one sorted singly linked list.

-------------------
use heap
1. put k list to the heap, comparing by head's value
2. pop the list from the heap:
   Add its head element to the return list.
   move the head to next element. If head is not none, add it to the heap.
3. repeat #2 until heap is empty.
   we can do some optimzation when the heap has only one elment.

time O(nlgk), n is the total number of element in the k list. space O(k)
"""


def question78():
    pass


def test_78():
    pass
