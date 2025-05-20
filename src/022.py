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

from collections import deque


def getOriginalWords(s: str, dictionary: list[str]) -> list[str]:
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
