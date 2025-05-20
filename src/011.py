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
    def __init__(self, isWord: bool = False):
        self.isWord = isWord
        self.children = {}


class TrieTree:
    def __init__(self):
        self.root = TrieNode()

    def add(self, query: str) -> None:
        node = self.root
        for c in query:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.isWord = True

    def queryWithPrefix(self, prefix: str) -> list[str]:
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

    def dfs(self, node: object, prefix: str, res: list[str]) -> None:
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
    assert tree.queryWithPrefix("de") == ["deer", "deal"]
