"""
question 24
Implement locking in a binary tree. A binary tree node can be locked or
unlocked only if all of its descendants or ancestors are not locked.

Design a binary tree node class with the following methods:

is_locked, which returns whether the node is locked
lock, which attempts to lock the node. If it cannot be locked, then it should
return false. Otherwise, it should lock it and return true.

unlock, which unlocks the node. If it cannot be unlocked, then it should
 return false. Otherwise, it should unlock it and return true.

You may augment the node to add parent pointers or any other property you
 would like. You may assume the class is used in a single-threaded program,
 so there is no need for actual locks or mutexes. Each method should run in
 O(h), where h is the height of the tree.
-------------------

Based on the description of the problem, the lock/unlock operation can be
called from any node but all its accestors or decendents needs to be in the
same status: unlocked or locked. That means the entire binary tree needs to
behavior consistently. The entire binary tree is either locked or unlocked.

Store the locked/unlocked status in the root node. When try to lock/unlock
from any node, travel to the root node and check the status and act accordingly.
We need to add parent pointer to the tree nodes, so traveling from any node to
the root node takes O(h) time.
"""


def test_24():
    pass
