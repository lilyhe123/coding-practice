"""
question 80
Given the root of a binary tree, return a deepest node. For example, in the
following tree, return d.

    a
   / \
  b   c
 /
d
-------------------
create a recursive function to travel the tree in depth-first traversal
and return a pair, first is the depth of the tree and second is the deepest node.

def dfs(tree, depth):
    if not tree:
        return (0, None)
    # leaf node
    if not tree.left and not tree.right:
        return (depth, tree)
    left = dfs(tree.left, depth+1)
    right = dfs(trree.right, depth+1)
    return left if left[0] > right[0] else right

"""


def question80():
    pass


def test_80():
    pass
