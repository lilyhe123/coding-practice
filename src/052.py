"""
question 52
Implement an LRU (Least Recently Used) cache. It should be able to be
initialized with a cache size n, and contain the following methods:

set(key, value): sets key to value. If there are already n items in the
cache and we are adding a new item, then it should also remove the least
recently used item.
get(key): gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.
-------------------

double linked list:  ordered in the recently used time. remove/add, O(1) time
  Node: key, value, pre, next
Hashmap<K, Integer>: key -> Node
get(): if exist, remove from the current place and add to
       the head of the queue
       if not exist, return null

set(): if exist, update its value and move it to the head of the list
       if not exist, add to the head of the list.
       if size > N, remove tail element
"""

from ds.classes import DoubleLinkedList


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.myMap = {}
        self.myList = DoubleLinkedList()

    def set(self, key, value):
        if key in self.myMap:
            # update existing element
            node = self.myMap[key]
            node.value = value
            self.myList.moveToHead(node)
        else:
            # create new element
            node = self.myList.add(key, value)
            self.myMap[key] = node
            # if cache exceeds its capacity, remove tail element
            if len(self.myMap) > self.capacity:
                node = self.myList.removeTail()
                self.myMap.pop(node.key)

    def get(self, key):
        if key in self.myMap:
            node = self.myMap[key]
            self.myList.moveToHead(node)
            return node.value
        else:
            return None

    def print(self):
        self.myList.print()


def test_52():
    cache = LRUCache(5)
    pairs = {"key1": 1, "key2": 2, "key3": 3, "key4": 4, "key5": 5, "key6": 6}
    for k, v in pairs.items():
        cache.set(k, v)
    cache.print()
    cache.get("key3")
    cache.print()
    cache.set("key4", 40)
    cache.print()


if __name__ == "__main__":
    test_52()
