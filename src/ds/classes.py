class LinkedNode:
    def __init__(self, val):
        self.val = val
        self.next = None

    def print(self):
        node = self
        s = ""
        while node:
            s += str(node.val)
            if node.next:
                s += "->"
            node = node.next
        print(s)

    @staticmethod
    def createLinkedList(nums):
        head, pre = None, None

        for num in nums:
            node = LinkedNode(num)
            if not head:
                head = node
                pre = node
            else:
                pre.next = node
            pre = node
        return head


class DoubleLinkedNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None
        # linkedList containing the node
        self.list = None


# !! use dummy head and tail to simplify code
class DoubleLinkedList:
    def __init__(self):
        self.head = DoubleLinkedNode(None, None)
        self.tail = DoubleLinkedNode(None, None)
        self.head.next = self.tail
        self.tail.pre = self.head

    def isEmpty(self):
        return self.head.next == self.tail

    # insert after the dummy head
    def addToHead(self, node):
        node.next = self.head.next
        node.pre = self.head
        self.head.next.pre = node
        self.head.next = node
        node.list = self

    def add(self, key, value) -> DoubleLinkedNode:
        node = DoubleLinkedNode(key, value)
        self.addToHead(node)
        return node

    def removeNode(self, node) -> None:
        # remove from current position
        node.pre.next = node.next
        node.next.pre = node.pre

    def moveToHead(self, node) -> None:
        if node.pre == self.head:
            return
        self.removeNode(node)
        self.addToHead(node)

    def removeHead(self) -> DoubleLinkedNode:
        if self.isEmpty():
            return
        node = self.head.next
        self.head.next = node.next
        node.next.pre = self.head
        return node

    def removeTail(self) -> DoubleLinkedNode:
        if self.isEmpty():
            return
        node = self.tail.pre
        node.pre.next = self.tail
        self.tail.pre = node.pre
        return node

    def print(self):
        node = self.head.next
        s = ""
        while node != self.tail:
            s += node.key + ": " + str(node.value)
            if node.next:
                s += ", "
            node = node.next
        print(s)


class TreeNode:
    def __init__(self, val, left: object = None, right: object = None):
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
