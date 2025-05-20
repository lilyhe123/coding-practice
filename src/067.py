"""
!! question 67
Implement an LFU (Least Frequently Used) cache. It should be able to be
initialized with a cache size n, and contain the following methods:

set(key, value): sets key to value. If there are already n items in the
cache and we are adding a new item, then it should also remove the least
frequently used item. If there is a tie,
then the least recently used key should be removed.

get(key): gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.
-------------------

When the cache is full, first try to remove the last frequenctly used item.
If there is a tie, reove the least recently used key

We need to record information of each key:
usage frequency and last used time

Node: key, value, freq, pre, next
find an item by key with constant time, we need a map by key

- key_map: key->Item
when the cache is full, we need to remove the LFU item.
To find it and remove it in constant time, we need a map by usage frequency
and seperate double linked list for each freq, the list is ordered by LUT.
- freq_map: freq -> double_list,
  the list is ordered by LUT, LRU node is on the tail
- minFreq: store the min frequency whose list are not empty

set:
    if key not exists:
     if the cache is full, remove the LFU node.
       find the list with minFreq, remove the tail of the list.
       remove the node from key_map.

     add the new node to key_map and freq_map with freq=1.
     update minFreq to 1.
    if key exists:
     find the node in key_map.  time O(1)
     update freq_map:
       remove from current list, with pre and next we can do it in constant time.
       the freq++ and add the item to the head of the list of the new freq.
    update minFreq:
       when remove for the list and list became empty, if minFreq is the freq, minFreq++.


get: find the node in key_map, return its value. time O(1)

"""

from ds import DoubleLinkedList, DoubleLinkedNode


# !! class inheritance
class LFUNode(DoubleLinkedNode):
    def __init__(self, key, value):
        super().__init__(key, value)
        self.freq = 1


class LFUCache:
    def __init__(self, n: int):
        self.n = n
        self.key_map = {}
        self.freq_map = {}
        self.minFreq = 1

    def size(self):
        return len(self.key_map)

    def set(self, key, val):
        # key not exists
        if key not in self.key_map:
            if len(self.key_map) == self.n:
                self.removeLFU()
            node = LFUNode(key, val)
            self.key_map[key] = node
            freq = 1
            if freq not in self.freq_map:
                self.freq_map[freq] = DoubleLinkedList()
            self.freq_map[freq].addToHead(node)
            self.minFreq = 1
        else:
            # key exists
            node = self.key_map[key]
            node.value = val
            # remove for current list
            node.list.removeNode(node)
            # update minFreq
            if node.list.isEmpty() and self.minFreq == node.freq:
                self.minFreq += 1
            # add to next freq's list
            node.freq += 1
            if node.freq not in self.freq_map:
                self.freq_map[node.freq] = DoubleLinkedList()
            self.freq_map[node.freq].addToHead(node)

    # no need to update minFreq since a new item is added when this method is called.
    # minFreq will be updated later
    def removeLFU(self):
        node = self.freq_map[self.minFreq].removeTail()
        if not node:
            raise ValueError("List of minFreq is empty. Remove fail.")
        del self.key_map[node.key]

    def get(self, key):
        if key in self.key_map:
            return self.key_map[key].value
        else:
            return None


def test_67():
    # test 1
    n = 3
    cache = LFUCache(n)
    assert cache.get("a") is None
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") == 1
    cache.set("d", 4)
    cache.set("e", 5)
    assert cache.size() == n
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.get("d") == 4
    assert cache.get("e") == 5

    # test 2
    n = 3
    cache = LFUCache(n)
    assert cache.get("a") is None
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # c.freq=1
    cache.set("a", 11)  # a.freq=2
    cache.set("b", 22)  # b.freq=2
    cache.set("d", 4)  # c is injected
    assert cache.get("c") is None
    cache.set("e", 5)  # d is injected
    assert cache.get("d") is None
    assert cache.get("a") == 11
    assert cache.get("b") == 22
    assert cache.get("e") == 5
    assert cache.size() == n

    # test 3
    n = 3
    cache = LFUCache(n)
    assert cache.get("a") is None
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.set("a", 11)  # a.freq=2
    cache.set("b", 22)  # b.freq=2
    cache.set("c", 33)  # c.freq=2
    # now minFreq is 2.
    cache.set("d", 4)
    assert cache.get("a") is None
    # now minFreq is 1
    cache.set("e", 5)
    assert cache.get("d") is None


test_67()
