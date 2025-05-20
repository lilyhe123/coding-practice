"""
question 26

Given a singly linked list and an integer k, remove the kth last element from
the list. k is guaranteed to be smaller than the length of the list.

The list is very long, so making more than one pass is prohibitively expensive.

Do this in constant space and in one pass.
-------------------

It's a single linked list, we can only travel from left to right follwing
the next pointer in each node.

Use two pointers to travel the linked list from left to right in one pass.
Initially p1 points to the head of the list, p2 points to the kth element of
the list. Then p1 and p2 go left along the list with the same pace,
one step at a time, until p2 reach the end of the list. p1 is pointing to the
kth last element of the list.
k = 3
a -> b -> c -> d -> e -> f -> g
^         ^
                    ^         ^
return 'e'
"""


def test_26():
    pass
