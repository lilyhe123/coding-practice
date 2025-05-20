"""
question 73
Given the head of a singly linked list, reverse it in-place.

-------------------
     A -> B -> C -> D -> E -> F
p    c
                              p    c

pre, cur = None, head
while cur:
  next = cur.next
  cur.next = pre
  pre = cur
  cur = next
return pre

"""


def test_73():
    pass
