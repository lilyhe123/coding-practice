class ListNode:
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
        self.head = ListNode(None, None)
        self.tail = ListNode(None, None)
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

    def add(self, key, value) -> ListNode:
        node = ListNode(key, value)
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

    def removeHead(self) -> ListNode:
        if self.isEmpty():
            return
        node = self.head.next
        self.head.next = node.next
        node.next.pre = self.head
        return node

    def removeTail(self) -> ListNode:
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
