"""
question 50
Suppose an arithmetic expression is given as a binary tree. Each leaf is
an integer and each internal node is one of '+', '−', '∗', or '/'.

Given the root to such a tree, write a function to evaluate it.

For example, given the following tree:

    *
   / \
  +    +
 / |  / \
3  2  4  5
You should return 45, as it is (3 + 2) * (4 + 5).
-------------------

dfs traversal, the recursive function returns the value of the subtree
!! how to define a class, how to use default parameters, whether you need
or need not specify the param name
"""


# !!! python is dynamic type, so we can use one class and one variable
# for number and operator
class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


def eval(node):
    if not node:
        return 0
    # process leaf node
    if not node.left and not node.right:
        return node.data
    # process non-leaf node
    val1 = eval(node.left)
    val2 = eval(node.right)
    if node.data == "+":
        return val1 + val2
    elif node.data == "-":
        return val1 - val2
    elif node.data == "*":
        return val1 * val2
    else:
        return val1 // val2


def test_50():
    tree = Node(
        "*",
        left=Node("+", left=Node(3), right=Node(2)),
        right=Node("+", left=Node(4), right=Node(5)),
    )
    print(eval(tree))
