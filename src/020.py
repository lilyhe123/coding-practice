"""
question 20
Given two singly linked lists that intersect at some point, find the
intersecting node. The lists are non-cyclical.

For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10,
return the node with value 8.

In this example, assume nodes with the same value are the exact
same node objects.

Do this in O(M + N) time (where M and N are the lengths of the lists)
and constant space.
-------------------

 3 <- 7 <- 8 <- 10

99 <- 1 <- 8 <- 10

    None <- 3 <- 7 <- 8 <- 10
                           p    c
pre = None
cur = head
while cur:
  next = cur.next
  cur.next = pre
  pre = cur
  cur = next
head1 = pre


1. brute-force: time O(N*M), space O(1)
2. use two stacks. time O(N+M), space O(N+M)
3. modify the linked lists
   use next pointer to point to its previous.
"""

from ds.classes import LinkedNode


def revert(head):
    pre, cur, next = None, head, None
    while cur:
        next = cur.next
        cur.next = pre
        pre = cur
        cur = next
    return pre


def getIntersectingNode(head1, head2):
    # edge case: if either list is None, there is no intersection
    if not head1 or not head2:
        return None
    # revers both linked lists
    head1 = revert(head1)
    head2 = revert(head2)
    # travel the list from right to left to find the intersaction
    node1, node2 = head1, head2
    intersectingNode = None
    while node1 and node2 and node1.val == node2.val:
        intersectingNode = node1
        node1 = node1.next
        node2 = node2.next
    # restore the original order of both linked list
    head1 = revert(head1)
    head2 = revert(head2)
    return intersectingNode


def runOneTest20(array_one, array_two, expected):
    head1 = LinkedNode.createLinkedList(array_one)
    head2 = LinkedNode.createLinkedList(array_two)
    if not expected:
        assert getIntersectingNode(head1, head2) is None
    else:
        assert getIntersectingNode(head1, head2).val == expected


def test_20():
    array_one = [3, 7, 8, 10]
    array_two = [99, 1, 8, 10]
    runOneTest20(array_one, array_two, 8)

    array_one = []
    array_two = [99, 1, 8, 10]
    runOneTest20(array_one, array_two, None)

    array_one = [1, 15, 20, 16, 10]
    array_two = [99, 1, 8, 10]
    runOneTest20(array_one, array_two, 10)

    array_one = [1, 15, 20, 16, 10]
    array_two = [1, 15, 20, 16, 10]
    runOneTest20(array_one, array_two, 1)


if __name__ == "__main__":
    test_20()
