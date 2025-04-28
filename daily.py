import random

import matplotlib.pyplot as plt

# uncomments the following line when need to debug stack overflow error
# import sys
# sys.setrecursionlimit(10)
"""
question 51
Given a function that generates perfectly random numbers between 1 and k
(inclusive), where k is an input, write a function that shuffles a deck
of cards represented as an array using only swaps.

It should run in O(N) time.

Hint: Make sure each one of the 52! permutations of the deck is
equally likely.
-------------------

input: random(k) return a peerfectrly random number between 1 and k,
k is an input.
contraints: using only swaps?
every one selection from i cards(i=52,51...1) should have equal probability
rand(52)*rand(51)
[1,52]
52, random[1,51];

"""


def shuffleCards():
    def randomGen(k):
        return random.randint(1, k)

    def swap(cards, i, j):
        tmp = cards[i]
        cards[i] = cards[j]
        cards[j] = tmp

    cards = [i for i in range(52)]
    for i in range(51, 0, -1):
        j = randomGen(i) - 1
        swap(cards, i, j)
    return cards


def test_51():
    print(shuffleCards())
    print(shuffleCards())


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


class ListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None


# !! use dummy head and tail to simplify code
class DoubleLinkedList:
    def __init__(self):
        self.head = ListNode(None, None)
        self.tail = ListNode(None, None)
        self.head.next = self.tail
        self.tail.pre = self.head

    # insert after the dummy head
    def addNode(self, node):
        node.next = self.head.next
        node.pre = self.head
        self.head.next.pre = node
        self.head.next = node

    def add(self, key, value) -> ListNode:
        node = ListNode(key, value)
        self.addNode(node)
        return node

    def moveToHead(self, node) -> None:
        if node.pre == self.head:
            return
        # remove from current position
        node.pre.next = node.next
        node.next.pre = node.pre
        self.addNode(node)

    def removeTail(self) -> ListNode:
        if self.head.next == self.tail:
            return None
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


"""
question 53
Implement a queue using two stacks. Recall that a queue is a FIFO
(first-in, first-out) data structure with the following methods: enqueue,
which inserts an element into the queue, and dequeue, which removes it.
-------------------

put: s1.push(e)
get: if s2 not empty: s2.pop
     else: push all elements from s1 to s2, the s2.pop
s1: 4->5->6

s2: 3->2->1
first is the top in s2, last is the top in s1
"""


class Queue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def enqueue(self, e):
        self.s1.append(e)

    def dequeue(self):
        if not self.s2:
            if not self.s1:
                return None
            while self.s1:
                self.s2.append(self.s1.pop(-1))
        return self.s2.pop(-1)

    def size(self):
        return len(self.s1) + len(self.s2)


def test_53():
    queue = Queue()
    nums = [1, 2, 3, 4]
    for num in nums:
        queue.enqueue(num)
    print(queue.dequeue())

    nums = [5, 6, 7, 8]
    for num in nums:
        queue.enqueue(num)
    while queue.size() > 0:
        print(queue.dequeue())


"""
question 54
Sudoku is a puzzle where you're given a partially-filled 9 by 9 grid with
digits. The objective is to fill the grid with the constraint that every row,
column, and box (3 by 3 subgrid) must contain all of the digits from 1 to 9.

Implement an efficient sudoku solver.
"""


def test_54():
    pass


"""
question 55 TODO
Implement a URL shortener with the following methods:

shorten(url), which shortens the url into a six-character alphanumeric string,
such as zLg6wl.
restore(short), which expands the shortened string into the original url.
If no such shortened string exists, return null.
Hint: What if we enter the same URL twice?
"""


def test_55():
    pass


"""
question 56
Given an undirected graph represented as an adjacency matrix and an integer k,
write a function to determine whether each vertex in the graph can be colored
such that no two adjacent vertices share the same color
using at most k colors.
-------------------

adjacent matrix: 0 means the two vertexes are adjacent with each other,
1 means not adjacent
[[0, 1, 1],
 [1, 0, 1],
 [1, 1, 0]]

1. Understand the problem
- input: adjacent matrix and int k
- constraint: no same color of two adjacent vertex
- output: true if the graph can be colored with k colors

2. Brainstorming
Som special cases
n is the total number of vertexes.
- If the graph is a complete graph, every pair of vertice are adjacent,
  we need at least n colors to color it.
- If the graph is a tree (a graph with no cycles). We can always color it
  with 2 colors. We travel the tree level by level and swap the color
  when going to next level.
- If k = 1, the graph must have no edge (no adjacent vertexes)
- If k >= n, always true.

Think about the brute-force approach. Every vertex can choose one from k
colors. We check all the possible combinations (with n loops) until we
find the one that doesn't violate the constraint. Every vertex has k
different options to choose and there are total n vertex, so the total
number of combinations are k^n.
Time complexity for burte-force approach is O(k^n).

How to optimize it?
Backtracking
We try to color vertices one by one. At each step choose one color for
the vertex without vilating the constraints. If at one step no feasible
color to the vertex, backtrack to previous steps and try other feasible
colors until we find a feasible color arrangement or exhaust all
possible combinations.

!! time complexity
within each recursive call, it decide the color for current vertex.
The recursive call travels a complete k-ary tree with n depths.
In the worst case it need to travel all nodes in the tree. The total
number of nodes in the tree is the same complexity of k^n.
So time complexity is O(k^n).

space complexity
- We create addtional data structure to store the candidate colors
  for each vertex. The colors list stores N elements and each element is
  a set with k size. So its space is O(nk)
- The recursion depth is n. so its space is O(n).
So space complexity is O(nk)
"""


def can_color(matrix, k):
    N = len(matrix)
    colors = [{1}]
    for i in range(1, N):
        candidates = {x for x in range(1, k + 1)}
        colors.append(candidates)
    # to store the color for each vertex
    rst = []

    def paint(index):
        if index == N:
            return True
        for selected in colors[index]:
            # choose one color
            rst.append(selected)
            # remove the selected color from its later adjacent vertexes
            for j in range(index + 1, N):
                if matrix[index][j] == 1:
                    colors[j].discard(selected)
            if paint(index + 1):
                return True
            rst.pop()
            # restore the color to its later adjacent vertexes
            for j in range(index + 1, N):
                if matrix[index][j] == 1:
                    colors[j].add(selected)
        return False

    isFeasible = paint(0)
    print(rst)
    return isFeasible


"""
The expected ourputs are:
[1, 2, 1]
True
[]
False
[1, 2, 3, 2]
True
"""


def test_56():
    matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    k = 2
    print(can_color(matrix, k))
    matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    k = 2
    print(can_color(matrix, k))
    matrix = [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
    k = 3
    print(can_color(matrix, k))


"""
question 57
Given a string s and an integer k, break up the string into multiple lines
such that each line has a length of k or less. You must break it up so that
words don't break across lines. Each line has to have the maximum possible
amount of words. If there's no way to break the text up, then return null.

You can assume that there are no spaces at the ends of the string and that
there is exactly one space between each word.

For example, given the string "the quick brown fox jumps over the lazy dog"
and k = 10, you should return:
["the quick", "brown fox", "jumps over", "the lazy", "dog"].
No string in the list has a length of more than 10.
-------------------

k = 10
"the quick brown fox jumps over the lazy dog"

tokens: ['the', 'quick', 'brown', 'fox']

time complexity:
- splitting the string into words, O(n), n is the length of the string.
- iterate words, O(m), m is the number of words
So overall time complexity is O(n+m).

space complexity:
We create data structure to store words and lines.
So space complexity is O(m).

"""


def break_str(s, k):
    tokens = s.split(" ")
    maxLen = max(len(x) for x in tokens)
    if maxLen > k:
        return None
    # line is always not empty
    line = tokens[0]
    line_list = []
    for i in range(1, len(tokens)):
        remainingLen = k - len(line) - 1
        if len(tokens[i]) <= remainingLen:
            line += " "
            line += tokens[i]
        else:
            line_list.append(line)
            line = tokens[i]
    # handle leftover
    line_list.append(line)
    return line_list


def test_57():
    s = "the quick brown fox jumps over the lazy dog"
    k = 10
    print(break_str(s, k))
    k = 4
    print(break_str(s, k))


"""
question 58
An sorted array of integers was rotated an unknown number of times.

Given such an array, find the index of the element in the array in faster
than linear time. If the element doesn't exist in the array, return null.

For example, given the array [13, 18, 25, 2, 8, 10] and the element 8,
return 4 (the index of 8 in the array).

You can assume all the integers in the array are unique.
-------------------

Try to use binary search to find the target value.

There is a turning point in the array.
for any subrange[left, right]
- if left > right, the turning point is included. The subarray is not sorted
- if left < right, the turning point is not included. The subarray is sorted.

So we can only rely on the sorted subarray to check whether the target
is in it or not. If yes, we keep search in this half,
otherwise we search on the other halp

start, mid, end
if target == nums[mid]: return mid
if nums[start] < nums[mid]:
  if nums[start] <= target < nums[mid]: end = mid - 1
  else: start = mid + 1
elif nums[mid] < nums[end]:
  if nums[mid] < target <= nums[end]: start = mid + 1
  else: end = mid - 1
"""


def findIndex(nums, target):
    def binarySearch(start, end):
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            if nums[start] <= nums[mid]:  # this is the sorted half
                if nums[start] <= target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
        return None

    return binarySearch(0, len(nums) - 1)


def test_58():
    nums = [13, 18, 25, 2, 8, 10]
    target = 8
    print(findIndex(nums, target))  # Output: 4
    target = 16
    print(findIndex(nums, target))  # Output: None


"""
question 59 TODO
Implement a file syncing algorithm for two computers over a low-bandwidth
network. What if we know the files in the two computers are mostly the same?
-------------------

This problem can be tackled with block-level syncing and delta encoding,
using well-established techniques from tools like rsync. The trick is
to minimize the data transfered while ensuring both computers
remain synchronized.

- Merkle tree algorithm
- rsync algorithm
MarkTree:
  construct by a hash list
  getHashByLevel(level)->List
"""
# from hashlib import md5


# MarkleTree is a binary tree, every node mostly has two child:
# left and right.
class MarkleNode:
    def __init__(self, hash, left=None, right=None):
        self.hash = hash
        self.left = left
        self.right = right

    def isLeaf(self):
        return not self.left and not self.right


class MarkleTree:
    def __init__(self, hash_list):
        self.root = self.construct(hash_list)

    def print(self):
        queue = [self.root]
        while queue:
            size = len(queue)
            s = ""
            for i in range(size):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                s += str(node.hash)
                s += " "
            print(s)

    def construct(self, hash_list):
        nodes = []
        i = 0
        # construt leave nodes
        for hash in hash_list:
            nodes.append(MarkleNode(hash))

        # construct nodes level-by-level bottom-up
        parents = []
        while len(nodes) > 1:
            i = 0
            while i < len(nodes):
                left = nodes[i]
                i += 1
                if i < len(nodes):
                    right = nodes[i]
                i += 1
                hash = left.hash + right.hash if right else left.hash
                parents.append(MarkleNode(hash, left, right))
            nodes = parents
            parents = []
        return nodes[0]


class File:
    def __init__(self, name, url=None, isLocal=True):
        self.name = name
        self.url = url
        self.isLocal = isLocal

    def getMarkleTree(self, hash_list):
        return MarkleTree(hash_list)


class FileSync:
    def __init__(self, local: File, remote: File):
        self.local = local
        self.remote = remote

    def sync(self):
        localTree = self.local.getMarkleTree([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        remoteTree = self.remote.getMarkleTree([1, 2, 13, 4, 5, 6, 7, 8, 9, 10])
        localTree.print()
        remoteTree.print()

        localNode = localTree.root
        remoteNode = remoteTree.root
        if localNode.hash == remoteNode.hash:
            return
        localNodes, remoteNodes = [localNode], [remoteNode]

        # keep compare and return different nodes until no more difference
        while localNodes or remoteNodes:
            # print(localNodes,remoteNodes)
            rst = self.compareAndSync(localNodes, remoteNodes)
            localNodes, remoteNodes = rst

    def compareAndSync(self, localNodes, remoteNodes):
        nextLocal, nextRemote = [], []

        for i in range(min(len(localNodes), len(remoteNodes))):
            localOne = localNodes[i]
            remoteOne = remoteNodes[i]
            if localOne.hash != remoteOne.hash:
                # find different node
                if not localOne.isLeaf():
                    nextLocal.append(localOne.left)
                    if localOne.right:
                        nextLocal.append(localOne.right)
                    nextRemote.append(remoteOne.left)
                    if remoteOne.right:
                        nextRemote.append(remoteOne.right)
                else:
                    # sync the block from remote to local
                    print("sync one block", localOne.hash, remoteOne.hash)

        if len(localNodes) > len(remoteNodes):
            if not localNodes[-1].isLeaf():
                nextLocal.append(localNodes[-1])
            else:
                print(
                    "find local last block is leftover, remove it?", localNodes[-1].hash
                )
        elif len(localNodes) < len(remoteNodes):
            if not remoteNodes[-1].isLeaf():
                nextRemote.append(remoteNodes[-1])
            else:
                print("sync remote last block", remoteNodes[-1].hash)

        return [nextLocal, nextRemote]


def test_59():
    fileSync = FileSync(File("file1"), File("file2"))
    fileSync.sync()


"""
question 60
Given a multiset of integers, return whether it can be partitioned into two
subsets whose sums are the same.

For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return
true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35},
which both add up to 55.

Given the multiset {15, 5, 20, 10, 35}, it would return false, since we
can't split it up into two subsets that add up to the same sum.
-------------------

target = 55
{15, 5, 20, 10, 35, 15, 10}
 ^
   0 1 2 3 4 5 6 7 8 9 10.   15.   20   55
0  y n n
15 y                          y
5. y         y                y.   y
20
change to problem to find a subset to a target value

1. brute-force: for every number, two options: choose it or not. Iterate all
combination and calculate the sum, if the sum is equal to the target, return
True. time O(2^n), space O(1).

2. recursion with memorization
  !! how to analyze the time and space complexity
  time O(n*m), space O(n*m), n is the size of given array and m is the half
  of the sum of the given array.

"""


def canPartition(nums):
    total = sum(nums)
    if total % 2 == 1:
        return False
    target = total // 2
    memo = set()

    def dfs(i, remaining):
        if remaining == 0:
            return True
        if target < 0 or i == len(nums) or (i, remaining) in memo:
            return False

        rst = dfs(i + 1, remaining) or dfs(i + 1, remaining - nums[i])
        if not rst:
            memo.add((i, remaining))
        return rst

    return dfs(0, target)


def test_60():
    nums = [15, 5, 20, 10, 35, 15, 10]
    assert canPartition(nums) is True
    nums = [15, 5, 20, 10, 35]
    assert canPartition(nums) is False


"""
question 61
Implement integer exponentiation. That is, implement the pow(x, y) function,
where x and y are integers and returns x^y.
Do this faster than the naive method of repeated multiplication.
For example, pow(2, 10) should return 1024.
-------------------

2*2...*2
x**y
x**y/2
10: 5+5
   (2+3)
   2*2+1

f(x, y):
  if y == 1: return x
  if y == 2: return x*x
  if y is even: f(x, y/2) * f(x, y/2)
  else: f(x, y/2) * f(x, y/2) * x

f(10): r * r-> f(5): v*v*x -> f(2):v
time O(lgy), space O(lgy)
"""


def pow(x, y):
    if y == 1:
        return x
    if y == 2:
        return x * x

    if y % 2 == 0:
        val = pow(x, y // 2)
        return val * val
    else:
        val = pow(x, y // 2)
        return val * val * x


def test_61():
    assert pow(2, 10) == 1024
    assert pow(2, 9) == 512


"""
question 62
There is an N by M matrix of zeroes. Given N and M, write a function to count
the number of ways of starting at the top-left corner and getting to the
bottom-right corner. You can only move right or down.

For example, given a 2 by 2 matrix, you should return 2, since there are two
ways to get to the bottom-right:

Right, then down
Down, then right
Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.
-------------------

f(n, m) = f(n, m-1) + f(n-1, m)
1 1 1  1  1
1 2 3  4  5
1 3 6  10 15
1 4 10 20 35
1 5 15 35 70

time O(n*m), space O(m)
"""


def totalWaysToMove(n, m):
    if n == 1:
        return 1
    pre = [1] * m
    cur = [0] * m
    for i in range(2, n + 1):
        cur[0] = 1
        for j in range(1, m):
            cur[j] = cur[j - 1] + pre[j]
        pre = cur
        cur = [0] * m
    return pre[m - 1]


def test_62():
    print("run tests62")
    assert totalWaysToMove(2, 2) == 2
    assert totalWaysToMove(5, 5) == 70


"""
question 63
Given a 2D matrix of characters and a target word, write a function that
returns whether the word can be found in the matrix by going left-to-right,
or up-to-down.

For example, given the following matrix:

[['F', 'A', 'C', 'I'],
 ['O', 'B', 'Q', 'P'],
 ['A', 'N', 'O', 'B'],
 ['M', 'A', 'S', 'S']]
and the target word 'FOAM', you should return true, since it's the
leftmost column. Similarly, given the target word 'MASS', you should return
true, since it's the last row.
-------------------

depth-first traversal in a matrix. time O(nm2^l)
"""


def wordExists(matrix, word):
    ROWS, COLS = len(matrix), len(matrix[0])

    def dfs(r, c, index):
        if matrix[r][c] != word[index]:
            return False
        if index == len(word) - 1:
            return True
        return (r + 1 < ROWS and dfs(r + 1, c, index + 1)) or (
            c + 1 < COLS and dfs(r, c + 1, index + 1)
        )

    for i in range(ROWS):
        for j in range(COLS):
            if dfs(i, j, 0):
                return True

    return False


def test_63():
    matrix = [
        ["F", "A", "C", "I"],
        ["O", "B", "Q", "P"],
        ["A", "N", "O", "B"],
        ["M", "A", "S", "S"],
    ]
    assert wordExists(matrix, "FOAM") is True
    assert wordExists(matrix, "MASS") is True
    assert wordExists(matrix, "ANAM") is False


"""
question 64
A knight's tour is a sequence of moves by a knight on a chessboard such that
all squares are visited once.
Given N, write a function to return the number of knight's tours on an
N by N chessboard.
-------------------

f(i,j):
steps: (1,2), (1,-2), (-1, 2), (-1, -2), (2,1), (2, -1), (-2, -1), (-2, 1)

- Exhaust all possible starting point, N*N options.
- Calculate number of tours for every start point.
  - DFS: at every step exaust all possible directions without visiting
    the same squars more than once. time O(8^(n*n))
- sum the numver of tours. time O(1)
"""


def countTours(N):
    steps = ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, -1), (-2, 1))
    TOTAL_COUNT = N * N
    visited = []
    visited = [[0 for i in range(N)] for j in range(N)]

    def dfs(i, j, visitedCount):
        if i < 0 or i >= N or j < 0 or j >= N or visited[i][j] > 0:
            return 0
        # go forward
        visited[i][j] = visitedCount
        count = 0
        if visitedCount == TOTAL_COUNT:
            # print(visited)
            count = 1
        else:
            for di, dj in steps:
                count += dfs(i + di, j + dj, visitedCount + 1)
        # backtracking
        visited[i][j] = 0
        return count

    tourCount = 0
    for r in range(N):
        for c in range(N):
            rst = dfs(r, c, 1)
            # print(r,c,rst)
            tourCount += rst
    return tourCount


def test_64():
    print("run test64")
    assert countTours(1) == 1
    assert countTours(2) == 0
    assert countTours(3) == 0
    assert countTours(4) == 0
    # assert countTours(5) == 1728


"""
question 65
Given a N by M matrix of numbers, print out the matrix in a clockwise spiral.

For example, given the following matrix:
[[1,  2,  3,  4,  5],
 [6,  7,  8,  9,  10],
 [11, 12, 13, 14, 15],
 [16, 17, 18, 19, 20]]
You should print out the following:
1,2,3,4,5,10,15,20,19,18,17,16,11,6,7,8,9,14,13,12
-------------------

directions: ((0,1), (1,0), (0, -1), (-1,0))
TOTAL = N*M
steps = []
row,col=0,0
count = 0
for dr,dc in directions:
  while row in range(N) and col in range(M) and matrix[row][col] > 1:
    steps.append(matrix[row][col])
    count += 1
    if count == TOTAL:
      return steps
    matrix[row][col] = 0
    row += dr
    col += dc

"""


def spinMatrix(matrix):
    N, M = len(matrix), len(matrix[0])
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    TOTAL = N * M
    steps = []
    row, col = 0, 0
    steps.append(matrix[0][0])
    matrix[0][0] = 0
    if TOTAL == 1:
        return steps
    while len(steps) < TOTAL:
        for dr, dc in directions:
            while (
                0 <= row + dr < N
                and 0 <= col + dc < M
                and matrix[row + dr][col + dc] > 0
            ):
                row += dr
                col += dc
                steps.append(matrix[row][col])
                matrix[row][col] = 0
    return steps


def test_65():
    print("run test65")
    matrix = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ]
    result = [1, 2, 3, 4, 5, 10, 15, 20, 19, 18, 17, 16, 11, 6, 7, 8, 9, 14, 13, 12]
    assert spinMatrix(matrix) == result


"""
question 66
Assume you have access to a function toss_biased() which returns 0 or 1 with
a probability that's not 50-50 (but also not 0-100 or 100-0).
You do not know the bias of the coin.

Write a function to simulate an unbiased coin toss.
-------------------

0:1: 3:7
0:1: 7:3
---------
0:1: 10:10
switch 0 and 1 every second time
"""


def toss_unbiased():
    def toss_biased():
        if random.randint(1, 10) <= 3:
            return 0
        else:
            return 1

    v1 = toss_biased()
    v2 = toss_biased()
    if v1 ^ v2:
        return v1
    else:
        return toss_unbiased()


def test_66():
    print("run test66")
    total = 99999
    c1, c2 = 0, 0
    nums = []
    for i in range(total):
        nums.append(toss_unbiased())
        if nums[-1]:
            c1 += 1
        else:
            c2 += 1
    diff = c1 / total * 100 - c2 / total * 100
    print(diff)
    assert -1 < diff < 1


"""
question 67 TODO
Implement an LFU (Least Frequently Used) cache. It should be able to be
initialized with a cache size n, and contain the following methods:

set(key, value): sets key to value. If there are already n items in the
cache and we are adding a new item, then it should also remove the least
frequently used item. If there is a tie,
then the least recently used key should be removed.

get(key): gets the value at key. If no such key exists, return null.
Each operation should run in O(1) time.
"""


def test_67():
    pass


def showHist():
    total = 9999
    nums = []
    bin = 10
    for i in range(total):
        nums.append(random.randint(1, bin))
    plt.title("Biased Distribution")
    plt.hist(nums, bin)
    plt.show()


# showHist()
