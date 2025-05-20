"""
question 8
A unival tree (which stands for "universal value") is a tree where
all nodes under it have the same value.
Given the root to a binary tree, count the number of unival subtrees.
For example, the following tree has 5 unival subtrees:
   0
  / \
 1   0
    / \
   1   0
  / \
 1   1
-------------------

first element means whether the tree is unival tree or not.
second element means the number of unival tree in this tree
 [True|False, int]

 f(node):
   if isLeaf: return [True, 0]

   if node.left:
     return [False, f(node.left)[1]]
   elif node.right:
     return [False, f(node.right)[1]]
   # both left and right not None

"""


class TreeNode:
    def __init__(self, val: int, left: object = None, right: object = None):
        self.val = val
        self.left = left
        self.right = right

    def print(self):
        print(self.val)

    def print_inorder(self):
        if self.left:
            self.print_inorder(self.left)
        self.print()
        if self.right:
            self.print_inorder(self.right)


def getUnivalTreeNum(node: object) -> int:
    def dfs(node: object) -> list:
        if not node:
            return [True, 0]
        # leaf node
        if not node.left and not node.right:
            return [True, 1]

        # handle one subtree is None.
        if not node.left:
            return [False, dfs(node.right)[1]]
        elif not node.right:
            return [False, dfs(node.left)[1]]

        # handle both subtrees are not None
        rtn = dfs(node.left)
        isLeftUnival, leftCount = rtn
        rtn = dfs(node.right)
        isRightUnival, rightCount = rtn
        total = leftCount + rightCount
        isUnival = False
        if isLeftUnival and isRightUnival and node.left.val == node.right.val:
            isUnival = True
            total += 1
        return [isUnival, total]

    return dfs(node)[1]


def test_8():
    root = TreeNode(
        0,
        left=TreeNode(1),
        right=TreeNode(
            0, left=TreeNode(1, left=TreeNode(1), right=TreeNode(1)), right=TreeNode(0)
        ),
    )
    assert getUnivalTreeNum(root) == 5
