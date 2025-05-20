"""
question 36
Given the root of a binary search tree, find the second largest
node in the tree.
-------------------

If we traverse the binary search tree in this order for every node:
node.right -> node -> node.left
the second node we visit is the second largest one.
We use a recursive function to do the traversal.
What's the base case? How we know we need to return?
We need to track how many nodes we already visited. If two nodes are
 already visited, return directly.
and we can store the visited noded to a input queue as an input parameter
"""


def findTheSecondLargest(node, visited):
    if not node:
        return
    if len(visited) == 2:
        return
    findTheSecondLargest(node.right, visited)
    if len(visited) == 2:
        return
    visited.append(node)
    if len(visited) == 2:
        return
    findTheSecondLargest(node.left, visited)


def test_36():
    pass
