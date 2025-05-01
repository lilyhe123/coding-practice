import heapq
import random
import time
from collections import deque
from threading import Thread
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# uncomments the following line when need to debug stack overflow error
# import sys
# sys.setrecursionlimit(10)

# 1 - 50
"""
!! question 1
Given a list of numbers and a number k, return whether any two numbers from
the list add up to k.
For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
Bonus: Can you do this in one pass?
-------------------

## Understand the problem
The input is a list of numbers and a number k, the output is a boolean...

## !! brain storming
1. brute-force approach
the most straight-forward approach is to find every possible pair of numbers
 in the list,
For every pair check whether the sum is equal to k
time O(n^2), space O(1)

2. sort the array and use two-pointers to find the pair that sum up to k.
3, 7, 10, 15
   ^
      ^
18 > 17, 13 < 17, 17 == 17 return true
Time O(nlgn), space O(1)

3. !!  to do this in one pass? This means we can only iterate the list once.
 We need an effective way to check whether we've already encountered the
 complement of the current number needed to reach K.
So we can introduce a hashset data structure to add all the already-visited
 numvers to it.
10, 15, 3, 7
7   2   14 10       ^
k=17
set: [10, 15, 3] return true
"""


def test_1():
    pass


"""
!! question 2
Given an array of integers, return a new array such that each element at index
 i of the new array is the product of all the numbers in the original array
 except the one at i.

For example, if our input was [1, 2, 3, 4, 5],
the expected output would be [120, 60, 40, 30, 24].
If our input was [3, 2, 1], the expected output would be [2, 3, 6].

Follow-up: what if you can't use division?
-------------------

## brainstorming
1. brute-force: for every number, calculate the product of all the
   other number.
   time O(n^2), space O(1)
2. in the brute force, we do many duplcated calculation:
   the product of the same numbers.
   if we can prevent the duplicated calculation we can optimize it.
   We can first calculate the product of all the numbers in the array.
   for every element, we only need to use the current nuber to divide
   the total product and get the result we need.
   time O(n), space O(1)
3. what if we can't use division?
  brute-force approach doesn't use division, but can we optimize it?

  formula for #2 is totalProduct/currentNumber
  new formula: product of numbers in its left * product of numbers
  in its right
  prefix and suffix products
  so we need two arrays, the first is the product of left numbers
  and the second is the product of right numbers.
  left_prod[i] = arr[0]*arr[1]..*arr[i-1]
  right_prod[i] = arr[i+1]* arr[i+2]...arr[n-1]
  output[i] = left_prod[i] * right_prod[i]

  !! optimze space - in-place calculation
  do the calculation directly in the result
  result[0] = 1
  i from 1 to n-1:
    result[i] = result[i-1] * nums[i-1]
  prod = 1
  i form n-2 to 0:
    prod *= nums[i+1]
    result[i] *= prod

   time O(n), space O(n)

  edge case: if there is a zero in the input array, the approach without
  division works correctly without specific handling this case.
"""


def test_2():
    pass


"""
question 3
Given the root to a binary tree, implement serialize(root),
which serializes the tree into a string,
and deserialize(s), which deserializes the string back into the tree.

For example, given the following Node class

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

The following test should pass:

node = Node('root', Node('left', Node('left.left')), Node('right'))
assert deserialize(serialize(node)).left.left.val == 'left.left'
"""


def test_3():
    pass


"""
question 4
Given an array of integers, find the first missing positive integer in linear
time and constant space. In other words, find the lowest positive integer
that does not exist in the array. The array can contain duplicates
and negative numbers as well.

For example, the input [3, 4, -1, 1] should give 2.
The input [1, 2, 0] should give 3.

You can modify the input array in-place.
-------------------

 1  2   3  4
 2 3 1 4
 1 2 3 4
[3, 4, -1, 1]
[1, -1, 3, 4]
for i in range():
  val = nums[i]
  if val in range(1, n+1):
    # swap elements in i and val+1
nums[i] = i+1
"""


def findLowest(nums):
    n = len(nums)
    for i in range(n):
        while nums[i] in range(1, n + 1) and i + 1 != nums[i]:
            # swap element in i and nums[i]-1
            j = nums[i] - 1
            tt = nums[i]
            nums[i] = nums[j]
            nums[j] = tt
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


def test_4():
    print(findLowest([3, 4, -1, 1]))
    print(findLowest([1, 2, 0]))


"""
question 5
cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the
first and last element of that pair. For example, car(cons(3, 4)) returns 3,
and cdr(cons(3, 4)) returns 4.

Given this implementation of cons:

def cons(a, b):
    def pair(f):
        return f(a, b)
    return pair
Implement car and cdr.
"""


def cons(a, b):
    def pair(f):
        return f(a, b)

    return pair


def car(pair):
    def getFirst(a, b):
        return a

    return pair(getFirst)


def cdr(pair):
    def getSecond(a, b):
        return b

    return pair(getSecond)


def test_5():
    print(car(cons(3, 4)))
    print(cdr(cons(3, 4)))


"""
question 6
An XOR linked list is a more memory efficient doubly linked list.
Instead of each node holding next and prev fields, it holds a field
named both, which is an XOR of the next node and the previous node.
Implement an XOR linked list; it has an add(element) which adds the
element to the end, and a get(index) which returns the node at index.

If using a language that has no pointers (such as Python), you can assume
you have access to get_pointer and dereference_pointer functions that
converts between nodes and memory addresses.

both: an XOR of the next node and the pre node.
how to get next node and pre node by both field?
"""


def test_6():
    pass


"""
question 7
Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count
 the number of ways it can be decoded.

For example, the message '111' would give 3, since it could be decoded
as 'aaa', 'ka', and 'ak'.

You can assume that the messages are decodable. For example,
'001' is not allowed.
-------------------

[1, 26] valid code
'1234'
  ^
f('n1n2...ni'):
  count = 0
  if int('n1') in range(1, 10):
    count += f('n2...ni')
  if int('n1n2') in range(10, 27):
    count += f('n3...ni')

dp = [0] * len(msg)
'd1d2d3....dn'

if d1>0: dp(0) = 1

if d2>0: dp(1) += dp(0)
if d1d2 in range(10, 27):
  dp(1) += 1

for i in range(2, len(msg)):
   if int(msg[i]) > 0:
     dp[i] += dp[i-1]
    if int(msg[i-1:i+1]) in range(10, 27):
      dp[i] += dp[i-2]
 return dp[len(msg)-1]
"""


def getDecodingWays(msg):
    dp0, dp1, dp2 = 1, 0, 0
    # '111' 1,2,3
    if int(msg[0]) > 0:
        dp1 = 1

    for i in range(1, len(msg)):
        if int(msg[i]) > 0:
            dp2 += dp1
        if int(msg[i - 1 : i + 1]) in range(10, 27):
            dp2 += dp0
        dp0 = dp1
        dp1 = dp2
        dp2 = 0
    return dp1


def test_7():
    print(getDecodingWays("111"))
    print(getDecodingWays("1111111"))


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
    def __init__(self, val, left=None, right=None):
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


def getUnivalTreeNum(node):
    def dfs(node):
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


"""
question 9
This problem was asked by Airbnb.

Given a list of integers, write a function that returns the largest sum of
non-adjacent numbers. Numbers can be 0 or negative.

For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5.
[5, 1, 1, 5] should return 10, since we pick 5 and 5.

Follow-up: Can you do this in O(N) time and constant space?
-------------------

Thinking
[2, 4, 6, 2, 5]
 ^
 contraints:
  - sum of non-adjacent numbers
 1. brute-force approach
iterate the array from the first to the last. For every element,
mostly there are two options: select it or not. If has chose the previous
element, then there is only one option to the current, not select.
time O(2^n), space O(1)

2. DP
dp[i]: the max sum of non-adjacent numbers ending in the ith element
2, 4, 6, 2, 5
2. 4, 8, 6, 13
dp[i] = max(dp[i-1], dp[i-2], dp[i-2]+nums[i])
"""


def maxSum4NonAdjacent(nums):
    dp0, dp1, dp2 = nums[0], max(nums[0], nums[1]), 0
    for i in range(2, len(nums)):
        dp2 = max(max(dp0, dp1), dp0 + nums[i])
        dp0 = dp1
        dp1 = dp2
    return dp1


def test_9():
    assert maxSum4NonAdjacent([2, 4, 6, 2, 5]) == 13
    assert maxSum4NonAdjacent([10, -1, 1, 2]) == 12


"""
question 10
Implement a job scheduler which takes in a function f and an integer n,
and calls f after n milliseconds.
-------------------

!! how to create and run a thread
"""


def scheduler():
    def schedule(func, n):
        time.sleep(n)
        func()

    def hello(name):
        print("hello " + name)

    job = Thread(target=schedule, args=(lambda: hello("from thread"), 2))
    job.start()

    print("Thread, you there?")
    time.sleep(3)
    print("Hello to you too")


def test_10():
    import doctest

    doctest.testmod(verbose=True)
    scheduler()


"""
!! question 11
Implement an autocomplete system. That is, given a query string s and a set
of all possible query strings, return all strings in the set that have s as
a prefix.

For example, given the query string de and the set of strings
[dog, deer, deal], return [deer, deal].

Hint: Try preprocessing the dictionary into a more efficient data structure
to speed up queries.
-------------------

input: a query string and a set of strings

1. brute-force
iterate through every string in the query_set.
to check whether the string starts with the prefix, we can user str.startwith
method. What the method will do is to compare the first m character.
If they are equal, add the string to the result list.
for every query, time O(N*M),
N is the size of the query_set, M is the size of the prefix.

2. optimization - preprocessing the dictionary
The presumption is that there are multiple queries.
To optimize the repeated queries with the same query_set,
we can preprocess the qeury_set into a data structure that
allow fast prefix-based search.

2.1 sort the query_set
- sort: time O(NlgN*L), N is the size of query_set, L is the average
  number of characters needs to compare for one pair of strings.
- query by prefix: time O(lgN*M)

2.2 Trie(prefix tree)
Trie is a tree-based data structure designed for efficient prefix-based
search. Every node in the trie represent a character and path from the root
 to the node form prefix in the dictionary.
- construct the trie tree with query_set. time(N*L),
  N is the size of the query_set,
  L is the length of the longest string in query_set.
- query on trie. time O(M + N1 * (L1-M)), N1 is the size of matched strings,
  L1 is the length of longest string in matched strings.

2.2 is more preferable if N is very large.

1. construct Trie tree with the given set of strings
class TrieNode:
  char: str, children: {str->TrieNode}
class TrieTree:
  root: TrieNode

  def add(s):
   node = root
   for c in s:
    if c not in node.children:
      node.children[c] = TrieNode()
    node = node.children[c]
"""


class TrieNode:
    def __init__(self, isWord=False):
        self.isWord = isWord
        self.children = {}


class TrieTree:
    def __init__(self):
        self.root = TrieNode()

    def add(self, query):
        node = self.root
        for c in query:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.isWord = True

    def queryWithPrefix(self, prefix):
        # find the node with the prefix
        node = self.root
        for c in prefix:
            if not node.children[c]:
                return None
            node = node.children[c]
        # get all str with the prefix
        res = []
        self.dfs(node, prefix, res)
        return res

    def dfs(self, node, prefix, res):
        if not node:
            return
        if node.isWord:
            res.append(prefix)
        for c, child in node.children.items():
            self.dfs(child, prefix + c, res)


def test_11():
    tree = TrieTree()
    queries = ["dog", "deer", "deal"]
    for s in queries:
        tree.add(s)
    print(tree.queryWithPrefix("de"))


"""
question 12
There exists a staircase with N steps, and you can climb up either 1 or 2
steps at a time. Given N, write a function that returns the number of unique
ways you can climb the staircase. The order of the steps matters.

For example, if N is 4, then there are 5 unique ways:
-------------------

1, 1, 1, 1
2, 1, 1
1, 2, 1
1, 1, 2
2, 2
What if, instead of being able to climb 1 or 2 steps at a time, you could
climb any number from a set of positive integers X? For example,
if X = {1, 3, 5}, you could climb 1, 3, or 5 steps at a time.

input: N steps of a stair case and a set for possible steps for each step.
output: the number of unique ways to clib the stari case.

N steps
X = {s1, s2...sm}, size of M
first let's try to find a way to break down th problem into smaller problem.

f(N, X) = f(N-s1, X) + f(N-s2, X)... + f(N-sm, X)

bottom-up
build up the solution from the base cases to the final value of N

1, 3, 5
  0 1 2 3 4 5 6 7 8
0 1 1 1 2 3

Notes:
- If order matters, dp is an array
  for dp[i], try to apply all the steps {s1,s2...sm}.
  formular dp[i] = dp[i-s1] + dp[i-s2] +...+dp[i-sm]
- If order doen't matter, it means different order with the same numbers
  only count once so only apply the steps one by one in sequence
  dp is a matrix, apply the {s1,s2...sm} one by one
"""


def totalWaysToClimb(n, steps):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(n + 1):
        for step in steps:
            if i >= step:
                dp[i] += dp[i - step]

    return dp[n]


def test_12():
    assert totalWaysToClimb(4, [1, 2]) == 5
    assert totalWaysToClimb(10, [1, 3, 5]) == 47


"""
question 13
Given an integer k and a string s, find the length of the longest substring
 that contains at most k distinct characters.

For example, given s = "abcba" and k = 2, the longest substring with k
distinct characters is "bcb".
-------------------

abcba
   i
    j
map: char->freq
if len(map) <= k: j++
else:
  find a local longest substr[i,j)
  move i forward until len(map) == k
"""


def longestSub(s, k):
    i = 0
    freqMap = {}
    maxLen = 1
    for j in range(0, len(s)):
        if s[j] not in freqMap:
            freqMap[s[j]] = 0
        freqMap[s[j]] += 1
        if len(freqMap) > k:
            maxLen = max(maxLen, j - i)
            while len(freqMap) > k:
                freqMap[s[i]] -= 1
                if freqMap[s[i]] == 0:
                    freqMap.pop(s[i])  # remove a key from map
                i += 1
    maxLen = max(maxLen, len(s) - i)
    return maxLen


def test_13():
    print(longestSub("abcba", 2))


"""
question 14
The area of a circle is defined as πr^2. Estimate π to 3 decimal
places using a Monte Carlo method.

Hint: The basic equation of a circle is x2 + y2 = r2.
-------------------

Monte Carlo Simulation is a type of computational algorithm that
uses repeated random sampling to obtain the likelihood of
a range of results of occurring.
"""


def test_14():
    pass


"""
question 15 TODO
Given a stream of elements too large to store in memory, pick a random
element from the stream with uniform probability.
-------------------

break down the problem
let's assume already pick an element randomly in [1, i].
Now i+1 element comes in, we need to pick up the i+1 element at 1/(i+1)
chance. so if random(1, i+1) == i+1, change to picked elment to t
he i+1 element. otherwise, the picked element remains no change.

reservoir sampling
if the stream totally have 5 elements:
probability to choose the 1th element: 1 * 1/2 * 2/3 * 3/4 * 4/5 = 1/5
probability to choose the 2th element:     1/2 * 2/3 * 3/4 * 4/5 = 1/5
probability to choose the 3th element:           1/3 * 3/4 * 4/5 = 1/5
probability to choose the 4th element:                 1/4 * 4/5 = 1/5
probability to choose the 5th element:                       1/5 = 1/5
"""


def getRandomFromStream(stream):
    rst = 0
    for i, val in enumerate(stream, 1):
        if random.randint(1, i + 1) == i:
            rst = val

    return rst


def create_hist(stream, no_of_samples=5000):
    hist = []
    for i in range(no_of_samples):
        val = getRandomFromStream(stream)
        hist.append(val)
    plt.hist(np.array(hist), align="left")
    plt.ylabel("Probability")
    plt.show()


def test_15():
    random.seed(1)
    # stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # create_hist(stream, 10000)


"""
question 16
You run an e-commerce website and want to record the last N order ids in a
log. Implement a data structure to accomplish this, with the following API:

record(order_id): adds the order_id to the log
get_last(i): gets the ith last element from the log.
i is guaranteed to be smaller than or equal to N.
You should be as efficient with time and space as possible.
-------------------

N=3
record: 1,2,3,4,5,6,5
[5,6,4]
record(i):
  if i in the list, move i to the head of the queue
  if i not in the list, add i to the head of the queue.
  If the len(queue) > N, remove the last element

list: []
"""


class OrderTracker:
    def __init__(self, N):
        self.N = N
        # id -> idx in queue
        self.queue = deque()

    def record(self, id):
        self.queue.append(id)
        if len(self.queue) > self.N:
            self.queue.popleft()

    def get_last(self, i):
        return self.queue[i]


def test_16():
    log = OrderTracker(3)
    ids = [1, 2, 3, 4, 5, 6]
    for id in ids:
        log.record(id)
    print(log.get_last(1))


"""
question 17
Suppose we represent our file system by a string in the following manner:
The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:

dir
    subdir1
    subdir2
        file.ext
The directory dir contains an empty sub-directory subdir1 and a sub-directory
subdir2 containing a file file.ext.

The string
"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" # noqa: E501
represents:
dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
The directory dir contains two sub-directories subdir1 and subdir2.
subdir1 contains a file file1.ext and an empty second-level sub-directory
subsubdir1. subdir2 contains a second-level sub-directory subsubdir2 containing
a file file2.ext.

We are interested in finding the longest (number of characters) absolute path
to a file within our file system. For example, in the second example above,
the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its length
is 32 (not including the double quotes).

Given a string representing the file system in the above format,
return the length of the longest absolute path to a file in the abstracted
file system. If there is no file in the system, return 0.
-------------------

longestLen = 0
tokens = str.split('/n')
paths = []
for token in tokens:
  if not paths: paths.append(token)
  else:
    count = token.sub('/t') # todo: how to count the substring
    token remove all /t # todo: how to remove all substrings
    paths = paths[:count]
    if isFile(token):
      size = len(token)
      for path in paths:
        size += len(path)
      size += len(paths)
      longestLen = max(longestLen, size)
    else:
      paths.append(token)
time O(n), space O(n), n is the token count in the given string
"""


def getLongestFilePath(s):
    longestPath = 0
    token_list = s.split("\n")
    path_list = []
    for token in token_list:
        count = token.count("\t")  # !! count the substring
        token = token.replace("\t", "")  # !! remove or replace all substrings
        path_list = path_list[:count]
        if token.endswith(".ext"):
            size = len(token)
            for path in path_list:
                size += len(path)
            size += len(path_list)
            longestPath = max(longestPath, size)
        else:
            path_list.append(token)
    return longestPath


def test_17():
    print("run test17")
    s = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
    assert getLongestFilePath(s) == 20
    s = "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"  # noqa: E501
    assert getLongestFilePath(s) == 32


"""
!! question 18
Given an array of integers and a number k, where 1 <= k <= length of the array,
 compute the maximum values of each subarray of length k.

For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get:
[10, 7, 8, 8], since:

10 = max(10, 5, 2)
7 = max(5, 2, 7)
8 = max(2, 7, 8)
8 = max(7, 8, 7)
Do this in O(n) time and O(k) space. You can modify the input array in-place
and you do not need to store the results. You can simply print them out
as you compute them.
-------------------

10, 5, 2, 7, 8, 7
   |      |
1. brute-force approach
time O((nk), space O(1)

2. use heap to get the max one with constant time
  max_heap with size of k,
  init: time O(klogk),
  window moving forward: remove the first one and add a new one, time O(k)
  time O(kn), space O(k)

3. deque to store indexes of elements in the current window but remove those
   useless (no chance to be the max) elements.
   If a larger element comes in, remove all the smaller elements in the queue.
   So elements storing in the queue are always in decreasing order.
    The first one in the queue is always the max number in the current window.
    time O(n), space O(k)
10, 5, 2, 7, 8, 7
"""


def getMaxInSubarray(nums, k):
    # the queue is to store indexes of elements in the currrent windoq
    queue = deque()
    output = []
    # initialize the queue with the first k numbers
    for i in range(k):
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop()
        queue.append(i)
    output.append(nums[queue[0]])
    # side the window across the array
    for start in range(1, len(nums) - k + 1):
        end = start + k - 1
        # remove indexes that are out of the window
        # actually since we move one space forward,
        # mostly we remove only one element
        if queue and queue[0] < start:
            queue.popleft()
        # add a new number to the end
        while queue and nums[queue[-1]] < nums[end]:
            queue.pop()
        queue.append(end)
        # first one in the queue is always the max number in the current window
        output.append(nums[queue[0]])
    return output


def test_18():
    print("run test18")
    nums = [10, 5, 2, 7, 8, 7]
    k = 3
    assert getMaxInSubarray(nums, k) == [10, 7, 8, 8]
    # increasing array
    nums = [1, 5, 10, 12, 18, 20, 21, 25, 29, 30]
    k = 5
    assert getMaxInSubarray(nums, k) == [18, 20, 21, 25, 29, 30]
    # decreasing array
    nums = [30, 29, 25, 21, 20, 18, 12, 10, 5, 1]
    k = 5
    assert getMaxInSubarray(nums, k) == [30, 29, 25, 21, 20, 18]


"""
question 19
A builder is looking to build a row of N houses that can be of K different
colors. He has a goal of minimizing cost while ensuring that no two
neighboring houses are of the same color.

Given an N by K matrix where the nth row and kth column represents
the cost to build the nth house with kth color, return the minimum cost
which achieves this goal.
If there is no way to build the houses, return -1.
-------------------

n=5, k=3
f(matrix, i, pre, memo):
  if i == N: return 0
  if memo[i][pre] == 0:
    minCost = float('inf')
    for j in range(k):
      if j != pre:
        minCost = min(minCost, matrix[i][j] + f(matrix, i+1, j))
    memo[i][pre] = minCost
  return memo[i][pre]
"""


def getMinCost(matrix):
    N, K = len(matrix), len(matrix[0])
    memo = [[float("inf")] * K for _ in range(N)]

    def dfs(i, preColor):
        if i == N:
            return 0
        if memo[i][preColor] == float("inf"):
            minCost = float("inf")
            for j in range(K):
                # for the first house, there is no prehouse hence no preColor
                if i == 0 or j != preColor:
                    minCost = min(minCost, matrix[i][j] + dfs(i + 1, j))
            memo[i][preColor] = minCost
        return memo[i][preColor]

    dfs(0, 0)
    return memo[0][0] if memo[0][0] != float("inf") else -1


def test_19():
    matrix = [[1, 2, 3, 4], [1, 2, 1, 0], [6, 1, 1, 5], [2, 3, 5, 5]]
    assert getMinCost(matrix) == 4  # Expected: 1,0,1,2
    matrix = [[1, 2, 3], [1, 2, 1], [3, 3, 1]]
    assert getMinCost(matrix) == 4  # Expected: 1,2,1

    matrix = [[3, 2, 4]]
    assert getMinCost(matrix) == 2
    # no way to build the houses
    matrix = [[5], [5], [5]]
    assert getMinCost(matrix) == -1


"""
question 20
Given two singly linked lists that intersect at some point, find the
intersecting node. The lists are non-cyclical.

For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10,
return the node with value 8.

In this example, assume nodes with the same value are the exact
same node objects.

Do this in O(M + N) time (where M and N are the lengths of the lists)
and constant space.
-------------------

 3 <- 7 <- 8 <- 10

99 <- 1 <- 8 <- 10

    None <- 3 <- 7 <- 8 <- 10
                           p    c
pre = None
cur = head
while cur:
  next = cur.next
  cur.next = pre
  pre = cur
  cur = next
head1 = pre


1. brute-force: time O(N*M), space O(1)
2. use two stacks. time O(N+M), space O(N+M)
3. modify the linked lists
   use next pointer to point to its previous.
"""


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


def revert(head):
    pre, cur, next = None, head, None
    while cur:
        next = cur.next
        cur.next = pre
        pre = cur
        cur = next
    return pre


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


def getIntersectingNode(head1, head2):
    # edge case: if either list is None, there is no intersection
    if not head1 or not head2:
        return None
    # revers both linked lists
    head1 = revert(head1)
    head2 = revert(head2)
    # travel the list from right to left to find the intersaction
    node1, node2 = head1, head2
    intersectingNode = None
    while node1 and node2 and node1.val == node2.val:
        intersectingNode = node1
        node1 = node1.next
        node2 = node2.next
    # restore the original order of both linked list
    head1 = revert(head1)
    head2 = revert(head2)
    return intersectingNode


def runOneTest20(array_one, array_two, expected):
    head1 = createLinkedList(array_one)
    head2 = createLinkedList(array_two)
    if not expected:
        assert getIntersectingNode(head1, head2) is None
    else:
        assert getIntersectingNode(head1, head2).val == expected


def test_20():
    array_one = [3, 7, 8, 10]
    array_two = [99, 1, 8, 10]
    runOneTest20(array_one, array_two, 8)

    array_one = []
    array_two = [99, 1, 8, 10]
    runOneTest20(array_one, array_two, None)

    array_one = [1, 15, 20, 16, 10]
    array_two = [99, 1, 8, 10]
    runOneTest20(array_one, array_two, 10)

    array_one = [1, 15, 20, 16, 10]
    array_two = [1, 15, 20, 16, 10]
    runOneTest20(array_one, array_two, 1)


"""
question 21
Given an array of time intervals (start, end) for classroom lectures
(possibly overlapping), find the minimum number of rooms required.
For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.
-------------------

(0, 50), (30, 75), (60, 150)
First sort the intervals by the start time and then scan the intervals
from the first to the last.
we need to keep tracking a group of latest overlapped intervals. When
the next interval comes in, how to decide some intervals nees to remove
from the group and which one to remove? This is the critical question
to this problem.
The answer is first-to-remove interval is the one with the minimum end time.
Compare the end time with the new start time, if the end time is no larger
than the start time then that interval need to be removed from the group.
Now we can see a heap is a proper data structure to hold the overlapped
intervals. It's a min_heap and the end time is to be compared in the heap.

-------------
   ------------
    --
        --------

min(end)>max(start)
"""


def getMinNumberOfRooms(intervals):
    # sort the intervals by start time
    intervals.sort(key=lambda x: x[0])
    maxSize = 1
    # create a min_heap compared by the end time,
    # track the maximum size of the heap
    hq = []
    for start, end in intervals:
        while hq and hq[0] <= start:
            heapq.heappop(hq)
        heapq.heappush(hq, end)
        maxSize = max(maxSize, len(hq))
    return maxSize


def test_21():
    intervals = [(30, 50), (0, 20), (60, 150)]
    assert getMinNumberOfRooms(intervals) == 1
    intervals = [(30, 75), (0, 50), (60, 150)]
    assert getMinNumberOfRooms(intervals) == 2
    intervals = [(30, 75), (20, 40), (0, 50), (60, 150)]
    assert getMinNumberOfRooms(intervals) == 3
    intervals = [(30, 75), (20, 40), (0, 50), (35, 150)]
    assert getMinNumberOfRooms(intervals) == 4


"""
question 22
Given a dictionary of words and a string made up of those words (no spaces),
return the original sentence in a list. If there is more than one possible
reconstruction, return any of them. If there is no possible reconstruction,
then return null.

For example, given the set of words 'quick', 'brown', 'the', 'fox', and the
string "thequickbrownfox", you should return ['the', 'quick', 'brown', 'fox'].

Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the string
 "bedbathandbeyond", return either ['bed', 'bath', 'and', 'beyond] or
 ['bedbath', 'and', 'beyond'].
-------------------

1. Recursion with backtracking
Recursion funtion f(index, words). Starting from the index point,
try to match words in the dictionary. It can have 0 to * matches.
Do depth-first traversal. If we can reach to the end of the string,
one solution is found. If there is no word match at some index,
we need to backtrack to try other paths.
This approach has exponential time complexity.

2. recursion with memo.
With the #1 approach, we might reach to the same index multiple times. Use a cache
to memorize the results to reduce duplicated computations.
time O(nm), n is the length of the string, m is the size of the dictionary.
space O(n) since the recursive depth is O(n).

3. BFS traversal
Use a queue to store to-be-traveled index. Intial value is [0].
Use a visited array to ensure every index only travel once.
time O(nm), space O(n)
"""


def getOriginalWords(s: str, dictionary: List[str]) -> List[str]:
    # queue store the list of index along the way
    queue = deque()
    queue.append([0])
    visited = [0] * len(s)
    visited[0] = 1
    while queue:
        path = queue.popleft()
        idx = path[-1]
        for word in dictionary:
            if s[idx:].startswith(word):
                nextIdx = idx + len(word)
                new_path = path[:]
                new_path.append(nextIdx)
                if nextIdx == len(s):
                    rtn_words = []
                    for i in range(1, len(new_path)):
                        rtn_words.append(s[new_path[i - 1] : new_path[i]])
                    return rtn_words
                if visited[nextIdx]:
                    continue
                visited[nextIdx] = 1
                queue.append(new_path)
    # no valid path
    return None


def test_22():
    s = "thequickbrownfox"
    words = ["quick", "brown", "the", "fox"]
    assert getOriginalWords(s, words) == ["the", "quick", "brown", "fox"]
    s = "bedbathandbeyond"
    words = ["bed", "bath", "bedbath", "and", "beyond"]
    assert getOriginalWords(s, words) == ["bedbath", "and", "beyond"]


"""
question 23
You are given an M by N matrix consisting of booleans that represents
a board. Each True boolean represents a wall. Each False boolean
represents a tile you can walk on.

Given this matrix, a start coordinate, and an end coordinate, return
the minimum number of steps required to reach the end coordinate from
the start. If there is no possible path, then return null. You can move up,
left, down, and right. You cannot move through walls.
You cannot wrap around the edges of the board.

For example, given the following board:

[[f, f, f, f],
[t, t, f, t],
[f, f, f, f],
[f, f, f, f]]
and start = (3, 0) (bottom left) and end = (0, 0) (top left),
the minimum number of steps required to reach the end is 7,
since we would need to go through (1, 2) because there is a wall everywhere
else on the second row.
-----------------

From starting point, try every possible paths until reaching the end point.
Both DFS and BFS work fine here.
But to find the minimum steps, we need to use BFS.
"""


def getMinSteps(matrix, starting, ending):
    ROLS, COLS = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    queue = deque()
    visited = [[0 for _ in range(COLS)] for _ in range(ROLS)]
    queue.append(starting)
    visited[starting[0]][starting[1]] = 1
    steps = 1
    while queue:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            # try all the directions to go forward
            for dr, dc in directions:
                newr, newc = r + dr, c + dc
                # filter out invalid/already-visited/wall cells
                if (
                    newr < 0
                    or newr >= ROLS
                    or newc < 0
                    or newc >= COLS
                    or visited[newr][newc]
                    or matrix[newr][newc] == "t"
                ):
                    continue
                if (newr, newc) == ending:
                    return steps
                visited[newr][newc] = 1
                queue.append((newr, newc))
        steps += 1
    return -1


def test_23():
    matrix = [
        ["f", "f", "f", "f"],
        ["t", "t", "f", "t"],
        ["f", "f", "f", "f"],
        ["f", "f", "f", "f"],
    ]
    assert getMinSteps(matrix, (3, 0), (0, 0)) == 7


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


"""
question 25
Implement regular expression matching with the following special characters:

. (period) which matches any single character
* (asterisk) which matches zero or more of the preceding element
That is, implement a function that takes in a string and a valid regular
expression and returns whether or not the string matches the regular expression.

For example, given the regular expression "ra." and the string "ray", your
function should return true. The same regular expression on the string
"raymond" should return false.

Given the regular expression ".*at" and the string "chat", your function should
return true. The same regular expression on the string "chats" should return false.
-------------------

1. recursion and backtracking
input: pattern, text
f(i1, i2):
for ., i1++, i2++
for *, i1++, i2: i2, i2+1,...i2+n
others: if pattern[i1] == text[i2]: i1++, i2++; otherwise return

2. In solution #1 we might calculate the same f(i1, i2) multiple times.
To reduce the duplication, use DP.

populate the DP matrix row by row, only process the up-right half
  c h s a t s a
c y n n n n n n
.   y y y y y y
*     y y y y y
t       n y n n
.         n y y

for any cell, except (0,0), two possible directions to reach here:
1. from up-left
2. from left, only when current char in the pattern is '.'.

time O(N*M), space O(M)
"""


def regexMatched(pattern: str, text: str) -> bool:
    N, M = len(pattern), len(text)
    if N > M:
        return False
    pre = [0] * M
    cur = [0] * M
    for row in range(0, N):
        p = pattern[row]
        for col in range(row, M):
            t = text[col]
            # reach from up-left
            if (col == 0 or pre[col - 1]) and (p == "*" or p == t or p == "."):
                cur[col] = 1
            # reach from left
            elif (col == 0 or cur[col - 1]) and p == ".":
                cur[col] = 1
        # no valid path in this row, no chance to match, fail early
        if sum(cur) == 0:
            return False
        pre = cur
        cur = [0] * M
    return pre[M - 1] == 1


def test_25():
    assert regexMatched("c.*t.", "chsatsa")
    assert regexMatched(".*at", "chat")
    assert regexMatched(".*at", "sdfdedat")
    assert not regexMatched("chi.", "chat")
    assert regexMatched("ch*.", "chatsesd")
    assert regexMatched("*.", "blablabla")
    assert regexMatched(".a*", "blablablaay")
    assert not regexMatched("*a.", "blablablaay")


"""
question 26

Given a singly linked list and an integer k, remove the kth last element from
the list. k is guaranteed to be smaller than the length of the list.

The list is very long, so making more than one pass is prohibitively expensive.

Do this in constant space and in one pass.
--------------------

It's a single linked list, we can only travel from left to right follwing
the next pointer in each node.

Use two pointers to travel the linked list from left to right in one pass.
Initially p1 points to the head of the list, p2 points to the kth element of
the list. Then p1 and p2 go left along the list with the same pace,
one step at a time, until p2 reach the end of the list. p1 is pointing to the
kth last element of the list.
k = 3
a -> b -> c -> d -> e -> f -> g
^         ^
                    ^         ^
return 'e'
"""


def test_26():
    pass


"""
question 27
Given a string of round, curly, and square open and closing brackets, return
whether the brackets are balanced (well-formed).

For example, given the string "([])[]({})", you should return true.

Given the string "([)]" or "((()", you should return false.
---------------------

Instintively, a stack can help use here but why?
To do this, iterate through the string:
For every opening, it needs to pair to a later unmatched closing. But considering
nesting scenarios, an opening can follow up another opening. So we need to reserve
unmatched openings somewhere. when a closing is comming, it needs to pair with
the most recent unmatched opening.
So a stack, as its Last-In-First-Out nature, is a good fit for this
Iterate through the string, process every bracket:
- if it's opening, push to the stack.
- if it's closing, pop from the stack (the stack should be non-empty) and pair them.
  - If not pair, return early.
- Until we complete all brackets in the string. If the stack is empty, it's balanced.
Otherwise, unbalanced
"""


def test_27():
    pass


"""
question 28
Write an algorithm to justify text. Given a sequence of words and an integer
line length k, return a list of strings which represents each line, fully
justified.

More specifically, you should have as many words as possible in each line. There
should be at least one space between each word. Pad extra spaces when necessary
so that each line has exactly length k. Spaces should be distributed as equally
as possible, with the extra spaces, if any, distributed starting from the left.

If you can only fit one word on a line, then you should pad the right-hand side
with spaces.

Each word is guaranteed not to be longer than k.

For example, given the list of words ["the", "quick", "brown", "fox", "jumps",
"over", "the", "lazy", "dog"] and k = 16, you should return the following:

["the  quick brown", # 1 extra space on the left
 "fox  jumps  over", # 2 extra spaces distributed evenly
 "the   lazy   dog"] # 4 extra spaces distributed evenly"

-------------------
iterate words, trying to fit in as much words as possible.
use two pointers, first pointer is the first word of current line and
second pointer is the word to check whether it can be fit in to current line.

["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
   i1
           i2
initially i1 = 0
cur_len = len(word[i1])
when add new word[i2], new length is cur_len + len(word[i2]) + 1
if cur_len is smaller or equal to k:
    move i2 forward.
otherwise:
    create a new line with words from i1 to i2-1;
        distribute the extra spaces evenly between words starting from left
        remaining_spaces = k - cur_len
        number of spaces in words: total_num = i2 - i1
        even space number: even_spaces = 1 + remaining_spaces // total_num
        extra_remaining: remaining_spaces % total_num
    move i1 to i2 as the first word  and move i2 forward
"""


def justify_text(words, k):
    # generate line with words [start, end)
    # cur_len is the length with words and one space between words
    def generate_one_line(start: int, end: int, cur_len: int):
        new_line = ""
        space_num = end - start - 1
        if space_num == 1:
            # only one word in the line
            new_line += words[start]
            new_line += " " * (k - len(words[start]))
            return new_line
        small_space_num = 1 + (k - cur_len) // (space_num)
        small_spaces = " " * small_space_num
        large_spaces = small_spaces + " "
        # print(f"'{large_spaces}'")
        large_space_num = (k - cur_len) % (space_num)
        new_line += words[i1]
        for i in range(1, space_num + 1):
            if i <= large_space_num:
                new_line += large_spaces
            else:
                new_line += small_spaces
            new_line += words[start + i]
        return new_line

    lines = []
    i1 = 0
    cur_len = len(words[0])
    for i2 in range(1, len(words)):
        if cur_len + len(words[i2]) + 1 > k:
            # needs to generate new line with words in [i1, i2)
            lines.append(generate_one_line(i1, i2, cur_len))
            i1 = i2
            cur_len = len(words[i1])
        else:
            cur_len = cur_len + len(words[i2]) + 1
    # process left_over words from i1 to the end
    lines.append(generate_one_line(i1, len(words), cur_len))
    return lines


def test_28():
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    k = 16
    assert justify_text(words, k) == [
        "the  quick brown",
        "fox  jumps  over",
        "the   lazy   dog",
    ]


"""
question 29
Run-length encoding is a fast and simple method of encoding strings. The basic
idea is to represent repeated successive characters as a single count and
character. For example, the string "AAAABBBCCDAA" would be encoded as
"4A3B2C1D2A".

Implement run-length encoding and decoding. You can assume the string to be
encoded have no digits and consists solely of alphabetic characters. You can
assume the string to be decoded is valid.

-------------------
"AAAABBBCCDAA"
     ^
       ^
42A3B2C
^

num = num * 10 + int(c)
"""


def run_length_encoding(text):
    i1 = 0
    encoded = ""
    c1 = text[0]
    for i2 in range(len(text)):
        c2 = text[i2]
        if c1 != c2:
            encoded += str(i2 - i1)
            encoded += c1
            i1 = i2
            c1 = c2
    # process left_over
    encoded += str(len(text) - i1)
    encoded += text[i1]
    return encoded


def run_length_decoding(encoded):
    num = 0
    text = ""
    for i2 in range(len(encoded)):
        c = encoded[i2]
        if "0" <= c <= "9":
            num = num * 10 + int(c)
        else:
            # decode sequence of one character
            text += c * num
            num = 0
    return text


def test_29():
    text = "AAAABBBCCDAA"
    assert run_length_decoding(run_length_encoding(text)) == text
    text = "AAAAAAAAAAAAAAAABBBCCDDDDDDDDDDDAA"
    assert run_length_decoding(run_length_encoding(text)) == text


"""
question 30
You are given an array of non-negative integers that represents a
two-dimensional elevation map where each element is unit-width wall and the
integer is the height. Suppose it will rain and all spots between two walls get
filled up.

Compute how many units of water remain trapped on the map in O(N) time and O(1)
space.

For example, given the input [2, 1, 2], we can hold 1 unit of water in the
middle.

Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2 in
the second, and 3 in the fourth index (we cannot hold 5 since it would run off
to the left), so we can trap 8 units of water.

-------------------
[2,1,2]
|   |
| | |
for ith element,
- left_max_wall is the max height in heights[0, i],
- right_max_wall is the max height in heights[i, N-1]
so the water height is min(left_max_wall, right_max_wall) - cur_wall_height

[2,1,2]
left_max_array: [2,2,2]
right_max_array:[2,2,2]


left_max_array = [0] * N
left_max_array[0] = heights[0]
for i in range(1, len(heights)):
    left_max_array[i] = max(left_max_array[i-1], heights[i])
right_max_array = [0] * N
right_max_array[N-1] = heights[N-1]
for i in range(N-2, -1, -1):
    right_max_array = max(right_max_array(i+1), heights[i])
time O(N), space O(N)
TODO how to optimize to time O(N), space O(1)
"""


def totalTrappedWater(heights):
    N = len(heights)
    left_max_array = [0] * N
    left_max_array[0] = heights[0]
    for i in range(1, N):
        left_max_array[i] = max(left_max_array[i - 1], heights[i])
    right_max_array = [0] * N
    right_max_array[N - 1] = heights[N - 1]
    for i in range(N - 2, -1, -1):
        right_max_array[i] = max(right_max_array[i + 1], heights[i])
    water = 0
    for i in range(N):
        water += min(left_max_array[i], right_max_array[i]) - heights[i]
    return water


def test_30():
    assert totalTrappedWater([2, 1, 2]) == 1
    assert totalTrappedWater([3, 0, 1, 3, 0, 5]) == 8


"""
question 31
The edit distance between two strings refers to the minimum number of character
insertions, deletions, and substitutions required to change one string to the
other. For example, the edit distance between “kitten” and “sitting” is three:
substitute the “k” for “s”, substitute the “e” for “i”, and append a “g”.

Given two strings, compute the edit distance between them.

-------------------
kitten
^
sitting
^

 if c1==c2:
    i1++ i2++
 else:
    a. delete: i1++ -> go right
    b. insert: i2++ -> go down
    c. replace: i1++ i2++ -> go right-down
    steps ++

DP bottom-up
  k i t t e n
s 1 2 3 4 5 6
i 2 1 2 3 4 5
t 3 2 1 2 3 4
t 4 3 2 1 2 3
i 5 4 3 2 2 3
n 6 5 4 3 3 2
g 7 6 5 4 4 3
"""


def editDistance(src, dest):
    init = 0 if src[0] == dest[0] else 1
    pre = list(range(init, init + len(src)))
    cur = [0] * len(src)
    for i in range(1, len(dest)):
        cur[0] = pre[0] + 1
        for j in range(1, len(src)):
            if src[j] == dest[i]:
                cur[j] = pre[j - 1]
            else:
                cur[j] = min(min(pre[j - 1], pre[j]), cur[j - 1]) + 1
        pre = cur
        cur = [0] * len(src)
    return pre[len(src) - 1]


def test_31():
    assert editDistance("kitten", "sitting") == 3
    assert editDistance("sigtten", "sitting") == 3


"""
question 32 TODO
This problem was asked by Jane Street.
Suppose you are given a table of currency exchange rates,
represented as a 2D array.
Determine whether there is a possible arbitrage: that is, whether
there is some sequence of trades you can make, starting with some amount
A of any currency, so that you can end up with some amount
greater than A of that currency.

There are no transaction costs and you can trade fractional quantities.

Recall algorithms that are used to detect specific types of cycles in
graphs based on edge weights. Two algorithms that are often useful for
problems involving weighted paths and cycles come to mind.
- Bellman-Ford Algorithm: This algorithm can detect negative cycles.
  If, after V-1 iterations (where V is the number of vertices),
  we can still relax an edge, it means there is a negative cycle in the graph.
- Floyd-Warshall Algorithm: This algorithm can find the shortest paths
  between all pairs of vertices. We can check the diagonal elements of the
  resulting distance matrix. If any diagonal element is negative, it indicates
  a negative cycle starting and ending at that vertex.
"""


def test_32():
    pass


"""
question 33
Compute the running median of a sequence of numbers.
That is, given a stream of numbers, print out the median of the list
so far on each new element.
Recall that the median of an even-numbered list is the average of
the two middle numbers.
input: [2, 1,   5, 7,   2, 0, 5]
output:[2, 1.5, 2, 3.5, 2, 2, 2]
-------------------

1. Understand the problem
what's the input and output?
Input is a stream of numbers, output is print the median of the list
so far on each new element.
any constrains?
numbers are in the stream, so it's can be very huge number
2. brainstorm

"""


def test_33():
    pass


"""
question 34
Given a string, find the palindrome that can be made by inserting the fewest
number of characters as possible anywhere in the word. If there is more than
one palindrome of minimum length that can be made, return the
lexicographically earliest one (the first one alphabetically).

For example, given the string "race", you should return "ecarace", since we
can add three letters to it (which is the smallest amount to make a palindrome)
. There are seven other palindromes that can be made from "race" by adding
three letters, but "ecarace" comes first alphabetically.

As another example, given the string "google", you should return "elgoogle".
-------------------

1. understand the problem
what's the input and output?
input is a string, output is s palindrome string after insert some
character to the input string.

what's the constraints?
can only inserting new chars to make output string palindrome
if multiple palindromes with the same length exists, return the
lexicographically earliest one.

2. brainstorm
first we need to find the palindrome subsequence with max length
top-down
f(arr, i, j):
  if i == j: return 1
  if i > j: return 0
  if arr[i] == arr[j]:
    return 2 + f(arr, i+1, j-1)
  else:
    return max(f(arr, i+1, j), f(arr, i, j-1))

bottom-up
start|end
  l e e o e
l 1 1 2 2 3
e   1 2 2 3
e.    1 1 3
o.      1 1
e.        1

leetcode
  ^.   ^
Output: 5
Explanation: Inserting 5 characters the string becomes "leetcodocteel".
"""


def minInsertions(s: str) -> int:
    n = len(s)
    dp = [[0 for _ in range(n)] for _ in range(n)]
    for step in range(n):
        i = 0
        while i + step < n:
            j = i + step
            if step == 0:
                dp[i][j] = 1
            elif step == 1:
                if s[i] == s[j]:
                    dp[i][j] = 2
                else:
                    dp[i][j] = 1
            else:
                if s[i] == s[j]:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
            i += 1

    return n - dp[0][n - 1]


def test_34():
    print(minInsertions("leetcode"))


"""
question 35
Given an array of strictly the characters 'R', 'G', and 'B', segregate the
values of the array so that all the Rs come first, the Gs come second, and t
he Bs come last. You can only swap elements of the array.

Do this in linear time and in-place.

For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should
become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].
-------------------

p1 is the for the next R,
p3 is for the next B,
p2 is the next value to check
['R', 'R', 'R', 'G', 'G', 'B', 'B']
            1         2
                           3
"""


def sortRGBArray(arr):
    def swap(arr, i, j):
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp

    p1, p2, p3 = 0, 0, len(arr) - 1

    while p2 <= p3:
        if arr[p2] == "R":
            swap(arr, p1, p2)
            p1 += 1
            p2 += 1
        elif arr[p2] == "B":
            swap(arr, p2, p3)
            p3 -= 1
        else:
            p2 += 1


def test_35():
    arr = ["G", "B", "R", "R", "B", "R", "G"]
    sortRGBArray(arr)
    print(arr)


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


"""
question 37
The power set of a set is the set of all its subsets. Write a function that,
given a set, generates its power set.

For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3},
{1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.

You may also use a list or array to represent a set.
-------------------

f(set, start):
  f(set,start+1) | {set[start], f(set, start + 1)}
"""


def getPowerSet(alist):
    def getSubset(alist, start):
        if start == len(alist) - 1:
            return [[alist[start]]]
        subList = getSubset(alist, start + 1)
        newList = [[alist[start]]]
        newList.extend(subList)
        for item in subList:
            newItem = [alist[start]]
            for one in item:
                newItem.append(one)
            newList.append(newItem)
        return newList

    return [] + getSubset(alist, 0)


def test_37():
    print(getPowerSet([1, 2, 3]))


def test_38():
    pass


"""
question 38
You have an N by N board. Write a function that, given N, returns the number
of possible arrangements of the board where N queens can be placed on the
board without threatening each other, i.e. no two queens share the same
row, column, or diagonal.
"""

"""
question 39
Conway's Game of Life takes place on an infinite two-dimensional board of
square cells. Each cell is either dead or alive, and at each tick,
the following rules apply:

Any live cell with less than two live neighbours dies.
Any live cell with two or three live neighbours remains living.
Any live cell with more than three live neighbours dies.
Any dead cell with exactly three live neighbours becomes a live cell.
A cell neighbours another cell if it is horizontally, vertically,
or diagonally adjacent.

Implement Conway's Game of Life. It should be able to be initialized with
a starting list of live cell coordinates and the number of steps it should
run for. Once initialized, it should print out the board state at each step.
Since it's an infinite board, print out only the relevant coordinates, i.e.
from the top-leftmost live cell to bottom-rightmost live cell.

You can represent a live cell with an asterisk (*) and
a dead cell with a dot (.).
-------------------

neighbours: 8
live neighbours:
 < 2: die
 2, 3: keep living
 > 3: die
 3: die -> living

die -> living
living -> die

 input: a starting list of live cell coordinates,
 the number of steps it should run for?
 live cell set: {(row, col)...}
 add live cells and neighbours of every live cell to a map: cell ->living|die
 iterate cells in the map, apply the rules which might result in changing
 its stat. put live cells to a new set
 auxiliary methods:
   getAllNeighbours(row, col)->List<(int, int)>:
   getLiveNeighbourCount(map, row, col)->int

"""


class GameOfLife:
    def __init__(self, liveCells: list, steps=0):
        self.live_cells = set(tuple(cell) for cell in liveCells)
        self.steps = steps

    def getAllNeighbours(self, cell):
        neighbours = []
        row, col = cell
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if (i, j) != cell:
                    neighbours.append((i, j))
        return neighbours

    def getLiveNeighbourCount(self, cell):
        row, col = cell
        count = 0
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if (i, j) != cell and (i, j) in self.live_cells:
                    count += 1
        return count

    def nextGen(self):
        next_live_cells = set()
        to_check_cells = set()
        for cell in self.live_cells:
            for neighbour in self.getAllNeighbours(cell):
                to_check_cells.add(neighbour)

        for cell in to_check_cells:
            count = self.getLiveNeighbourCount(cell)
            # remain living
            if (count == 2 or count == 3) and cell in self.live_cells:
                next_live_cells.add(cell)
            # from die to living
            if count == 3 and cell not in self.live_cells:
                next_live_cells.add(cell)
        self.live_cells = next_live_cells

    def run(self):
        for _ in range(self.steps):
            print(self.live_cells)
            self.nextGen()


def test_39():
    # Example 1: Blinker pattern
    print("Blinker pattern")
    initial_blinker = [(0, 1), (1, 1), (2, 1)]
    blinker_pattern = GameOfLife(initial_blinker, 5)
    blinker_pattern.run()

    # Example 2: Toad pattern
    print("Toad pattern")
    initial_toad = [(1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)]
    toad_pattern = GameOfLife(initial_toad, 5)
    toad_pattern.run()

    # Example 3: Glider pattern
    print("Glider pattern")
    initial_glider = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    glider_pattern = GameOfLife(initial_glider, 10)
    glider_pattern.run()


def test_40():
    pass


"""
question 40
Given an array of integers where every integer occurs three times except
for one integer, which only occurs once, find and return the
non-duplicated integer.
For example, given [6, 1, 3, 3, 3, 6, 6], return 1.
Given [13, 19, 13, 13], return 19.
Do this in O(N) time and O(1) space.
-------------------

input: an integer array, every integer occures 3 times except for one
integer, which only occurs once
output: return the non-duplicated integer.
[6, 1, 3, 3, 3, 6, 6]
 ^  ^
 use bit manipulation. XOR etc
"""

"""
question 41
Given an unordered list of flights taken by someone, each represented
as (origin, destination) pairs, and a starting airport, compute the person's
itinerary. If no such itinerary exists, return null.
If there are multiple possible itineraries, return the lexicographically
smallest one.
All flights must be used in the itinerary.

For example, given the list of flights
[('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')]
and starting airport 'YUL',
you should return the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].

Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting
 airport 'COM', you should return null.

Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
and starting airport 'A',
 you should return the list ['A', 'B', 'C', 'A', 'C'] even though
 ['A', 'C', 'A', 'B', 'C'] is also a valid
 itinerary. However, the first one is lexicographically smaller.
-------------------

A: [B, C]
B: [C]
C: [A]
{'A' -> [['B', False], ['C', False]], 'B' -> [['C', False]], 'C' -> [['A', False]]} # noqa: E501

f(adjList, startAirport, remainingFlightsNum, list):
  if remainingFlightsNum == 0:
    return list
  for nextAirport, visited in enumerate(adjlist[startAirport]):
    if visited: continue

    rtn = f(adjList, nextAirport, remainingFlightsNum-1, list.add(nextAirport)): # noqa: E501
    if rtn: return rtn
  return None

use tuple as the hashkey in a visited map
"""


def smallestItinerary(flights: List[List[str]], start: str) -> List[str]:
    # construct the graph as adjList
    amap = {}
    visited = {}  # (src, dest) -> 0
    for flight in flights:
        src, dest = flight
        visited[(src, dest)] = 0
        if src not in amap:
            amap[src] = []
        amap[src].append(dest)
    for _, value in amap.items():
        value.sort()

    def dfs(amap, visited, start, rtnList) -> List[str]:
        if len(rtnList) == len(visited) + 1:
            return rtnList

        for next in amap[start]:
            if not visited[(start, next)]:
                visited[(start, next)] -= 1
                rtnList.append(next)
                rtn = dfs(amap, visited, next, rtnList)
                if rtn:
                    return rtn
                rtnList.pop(-1)
                visited[(start, next)] += 1
        return None

    return dfs(amap, visited, start, [start])


def test_41():
    print(smallestItinerary([["A", "B"], ["A", "C"], ["B", "C"], ["C", "A"]], "A"))


def test_42():
    pass


"""
question 42
Given a list of integers S and a target number k, write a function that
returns a subset of S that adds up to k. If such a subset cannot be made,
then return null.

Integers can appear more than once in the list. You may assume all numbers
in the list are positive.

For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1]
since it sums up to 24.
-------------------

[12, 1, 61, 5, 9, 2]
  ^

return a set of possible sums that is not greater than k
f(aset, start):
 subset = f(aset, sart+1)
 newset = set(subset)
 for arr in subset:
   if sum(arr) + k <= k:
     newset.add
recursion

f([12, 1, 61, 5, 9, 2], 24)
              ^
  f(0, 24, []) or f(0, 12, [12])
    f(1, 24, []) f(1, 23, [1]), f(1, 12, [12]), f(1, 11, [12, 1])
      f(2, 24, []),  f(2, 23, [1]), f(2, 12, [12]), f(2, 11, [12, 1])

top-down:
f(n, target, list)
  if target == 0: return list
  if target < 0: return None
  rtn = f(n+1, target, list)
  if rtn: return rtn
  rtn = f(n+1, target - n, list + nums[n])
  return rtn

bottom-up: (time O(n^2), space O(n^2))
dp array is subset list ended in ith element
for dp[i], iterate dp[0] to dp[i-1], for every subset, add a new item
if total sum < K, if sum == k, return
[12, 1, 61, 5, 9, 2]
               ^
{[12]}, {[12,1], [1]}, None, {[12, 5], [12,1,5], {1, 5}}, {[12,], []}
"""


def test_43():
    pass


"""
question 43
Implement a stack that has the following methods:

push(val), which pushes an element onto the stack
pop(), which pops off and returns the topmost element of the stack.
If there are no elements in the stack, then it should throw
an error or return null.
max(), which returns the maximum value in the stack currently.
If there are no elements in the stack, then it should throw an
error or return null.
Each method should run in constant time.
"""

"""
question 44
We can determine how "out of order" an array A is by counting the number
of inversions it has.
Two elements A[i] and A[j] form an inversion if A[i] > A[j] but i < j.
That is, a smaller element appears after a larger element.
Given an array, count the number of inversions it has. Do this faster
than O(N^2) time.
You may assume each element in the array is distinct.

For example, a sorted list has zero inversions.
The array [2, 4, 1, 3, 5] has three inversions:
(2, 1), (4, 1), and (4, 3).
The array [5, 4, 3, 2, 1] has ten inversions: every distinct pair
forms an inversion.
-------------------

total pair count: n(n-1)/2=10

2, 4, 1, 3, 5
2, 4
^
1, 3, 5
^
merge sort
first half and second half

f(nums, start, end):

  mid =
  f(nums, start, mid)
  f(nums, mid+1, end)
  # merge
  i, j
  if nums[i] > nums[j]:
    count += (mid - i + 1)
"""


def inversionCount(nums):
    def mergeSort(nums, start, end, arr):
        if start == end or start > end:
            return 0

        # / true division, return a float; // floor division,
        # return an integer
        # !!! use // to return an integer
        mid = start + (end - start) // 2
        count = 0
        count += mergeSort(nums, start, mid, arr)
        count += mergeSort(nums, mid + 1, end, arr)
        # merge two subarray to a new array: [start, mid], [mid+1, end]
        i, j = start, mid + 1
        index = start
        while index <= end:
            if i > mid or (j <= end and nums[i] > nums[j]):
                arr[index] = nums[j]
                if i <= mid:
                    count += mid - i + 1
                j += 1
            else:
                arr[index] = nums[i]
                i += 1
            index += 1
        # copy the values in new arrays to nums
        for i in range(start, end):
            nums[i] = arr[i]
        return count

    arr = [0] * len(nums)
    return mergeSort(nums, 0, len(nums) - 1, arr)


def test_44():
    print(inversionCount([1, 2, 3, 4, 5]))
    print(inversionCount([2, 4, 1, 3, 5]))
    print(inversionCount([5, 4, 3, 2, 1]))


# problem #45
"""
Using a function rand5() that returns an integer from 1 to 5 (inclusive)
with uniform probability,
implement a function rand7() that returns an integer from 1 to 7 (inclusive).

rand5()  [1, 5):  1, 2, 3, 4, 5
rand7() [1, 7):  1, 2, 3, 4, 5, 6, 7

Hints:
- expend the range: how you can use mulitple call of ran5() to generate a
  number in a range larger than [1,5]
- uniform probability is the key. It means each number in the range has an
  equal chance to be returned.
- rejection sampling: If you only want to use  a sub-range of a range,
  when the outcome falls within the sub-range, you can go ahead and use it.
  If it falls outside the sub-range, you can discard it and try generating
  a new outcome.

consider this:
- How can you combine the results of two rand5() calls (let's say the
  results are x and y, both between 1 and 5) to get a number in a range
  larger than 5?
- Once you have the larger range, can you identify a sub-range within it
  that is a multiple of 7 that you can use for your mapping? Think about
  using modulo operator. But be careful about the the starting value
  and the distribution

a
11, 12, 13, 14, 15
21, 22, 23, 24, 25
31, 32, 33, 34, 35
41, 42, 43, 44, 45
51, 52, 53, 54, 55

b
1,  2,  3,  4,  5
6,  7,  8,  9,  10
11, 12, 13, 14, 15
16, 17, 18, 19, 20
21, 22, 23, 24, 25

random [1, 25]
base 5
c = rand5() * 5 + rand5() - 5

# if in the range [1, 21], module with 7
if c <= 21
  return c % 7 + 1
else: try again
random [1, 7]
"""


def rand7Again():
    # generate uniform random number in [0, 24]
    sum = rand5() * 5 + rand5() - 6
    # [0, 6], [7, 13], [14, 20]
    if sum < 21:
        #
        return sum % 7 + 1
    else:
        return rand7Again()


def rand7():
    def rand5():
        return random.randint(1, 5)

    # generate uniform random number in [1, 25]
    sum = rand5() * 5 + rand5() - 5
    if sum < 22:
        #
        return sum % 7 + 1
    else:
        return rand7()


def test_45():
    total = 1000000
    map = {}
    for i in range(total):
        val = rand7()
        if val not in map:
            map[val] = 0
        map[val] += 1

    alist = list(map.keys())
    alist.sort()
    for num in alist:
        print(num, map[num] / total * 100)


# problem #71
"""
Using a function rand7() that returns an iterger form 1 to 7 (inclusive)
with uniform probability,
implement a function rand5() that returns an integer from 1to 5 (inclusive).
"""


def rand5():
    def rand7():
        return random.randint(1, 7)

    # generate a number in range [1, 49]
    sum = rand7() * 7 + rand7() - 7

    # [1, 45] is the multiple range of [1, 5]. If sum is in range [1, 45],
    # use it to map to [1, 5] via modulo operator. If not, try again
    if sum < 46:
        return sum % 5 + 1
    else:
        return rand5()


def test_71():
    total = 5000000
    map = {}
    for i in range(total):
        val = rand5()
        if val not in map:
            map[val] = 0
        map[val] += 1

    alist = list(map.keys())
    alist.sort()
    for num in alist:
        print(num, map[num] / total * 100)


"""
question 46
Given a string, find the longest palindromic contiguous substring.
If there are more than one with the maximum length, return any one.
For example, the longest palindromic substring of "aabcdcb" is "bcdcb".
The longest palindromic substring of "bananas" is "anana".
-------------------

1. brute-force approach
check every possible substring from length n to length 2.
total number of substring is n^2. To check whether a substring is
palindromic, take linear time.
time O(n^3), space O(1).

2. optimize
bottom-up DP: time O(n^2), space O(n^2)
f(str, start, end):
   if start == end: return True
   if start == end - 1: return str[start] == str[end]

  if str[start] == str[end] && f(str, start+1, end-1): return True
  if str[start] != str[end]: return False

len(substring): 1 to n
bananas
start|end b a n a n a s
       b  y n n
       a.   y n y
       n      y.n y
       a.       y n y
       n.         y n
       a.           y n
       s              y
"""


def getLongestPalindrome(s):
    def isPalindrome(s, start, end, memo):
        if start >= end:
            return True
        key = (start, end)
        if key not in memo:
            if s[start] == s[end] and isPalindrome(s, start + 1, end - 1, memo):
                memo[key] = 1
            else:
                memo[key] = 0

        return memo[key] == 1

    memo = {}  # (i,j) -> 0|1 false|true
    for length in range(len(s) - 1, 0, -1):
        i = 0
        while i + length < len(s):
            if isPalindrome(s, i, i + length, memo):
                return s[i : i + length + 1]
            i += 1
    print(memo)
    return s[0:1]


def test_46():
    print(getLongestPalindrome("bananas"))


"""
question 47
Given a array of numbers representing the stock prices of a company in
chronological order, write a function that calculates the maximum profit
you could have made from buying and selling that stock once.
You must buy before you can sell it.

For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you
could buy the stock at 5 dollars and sell it at 10 dollars.
-------------------

1. brute-force
check every possible pair and return the max diff. time O(n^2), space O(1)

2. iterate the array and keep tracking smallest number in previous.
time O(n), space O(1)
"""


def getMaxProfit(nums):
    pre_smallest = nums[0]
    diff = 0
    for i in range(1, len(nums)):
        if nums[i] > pre_smallest:
            diff = max(diff, nums[i] - pre_smallest)
        else:
            pre_smallest = nums[i]
    return diff


def test_47():
    print(getMaxProfit([9, 11, 8, 5, 7, 10]))


"""
question 48
Given pre-order and in-order traversals of a binary tree,
write a function to reconstruct the tree.

For example, given the following preorder traversal:

[a, b, d, e, c, f, g]

And the following inorder traversal:

[d, b, e, a, f, c, g]

You should return the following tree:

    a
   /  \
  b    c
 / |  / \
d  e  f  g
"""


def test_48():
    pass


"""
question 49
Given an array of numbers, find the maximum sum of any contiguous
subarray of the array.

For example, given the array [34, -50, 42, 14, -5, 86],
the maximum sum would be 137,
since we would take elements 42, 14, -5, and 86.

Given the array [-5, -1, -8, -9], the maximum sum would be 0,
since we would not take any elements.

Do this in O(N) time.
-------------------

[34, -50, 42, 14, -5, 86]
      ^
"""


def maxSum(nums):
    sum = 0
    maxSum = 0
    for num in nums:
        sum += num
        if sum <= 0:
            sum = 0
        maxSum = max(maxSum, sum)
    return maxSum


def test_49():
    print(maxSum([34, -50, 42, 14, -5, 86]))
    print(maxSum([-5, -1, -8, -9]))


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
