"""
question 72
In a directed graph, each node is assigned an uppercase letter. We define a
path's value as the number of most frequently-occurring letter along that path.
For example, if a path in the graph goes through "ABACA", the value of the path
is 3, since there are 3 occurrences of 'A' on the path.

Given a graph with n nodes and m directed edges, return the largest value path
of the graph. If the largest value is infinite, then return null.

The graph is represented with a string and an edge list. The i-th character
represents the uppercase letter of the i-th node. Each tuple in the edge list
(i, j) means there is a directed edge from the i-th node to the j-th node.
Self-edges are possible, as well as multi-edges.

For example, the following input graph:
ABACA
[(0, 1),
(0, 2),
(2, 3),
(3, 4)]
Would have maximum value 3 using the path of vertices [0, 2, 3, 4], (A, A, C,
A).

The following input graph:
A
[(0, 0)]
Should return null, since we have an infinite loop.

-------------------
In a graph the largest value path should be in one of the paths starting
from nodes with no precedent node.
So starting from one of the nodes, do a dfs or bfs traversal to the graph,
if detecting a cycle return None. If no cycle, it's a tree. we calculate
all the path from the root node to the leaf node, generate the str along
each path and calulate the value and return the largest value.

edge cases: self loop, no node with no prenodes

First with the given input, we do some transformation.
precount list: for each node, store number of precedent nodes for each node.
next nodes list: for each node, store list of next nodes.

time O(n+m), space O(n+m)
"""


def calculateValue(path):
    freq_map = {}
    for c in path:
        if c not in freq_map:
            freq_map[c] = 0
        freq_map[c] += 1
    return max(freq_map.values())


def findLargestValuePath(nodes: str, edges: list):
    # transform the input
    N = len(nodes)
    precounts = [0] * N
    nextNodes = [[] for _ in range(N)]
    for fromNode, toNode in edges:
        if fromNode == toNode:
            # self loop
            return None
        precounts[toNode] += 1
        nextNodes[fromNode].append(toNode)
    # find node with precount = 0
    roots = []
    for i, count in enumerate(precounts):
        if count == 0:
            roots.append(i)

    if len(roots) == 0:
        return None

    # return True if a cycle detected
    def dfs(
        i: int, path: list[str], visited: list[bool], largestValue: list[int]
    ) -> bool:
        # visit the same node twice in one path, a cycle is detected
        if visited[i]:
            return True
        visited[i] = True
        path.append(nodes[i])
        if not nextNodes[i]:
            # reach a leaf node
            val = calculateValue(path)
            largestValue[0] = max(largestValue[0], val)
        else:
            for next in nextNodes[i]:
                if dfs(next, path, visited, largestValue):
                    return True
        visited[i] = False
        path.pop(-1)
        return False

    largestValue = [0]
    path = []
    visited = [False] * N
    for root in roots:
        if dfs(root, path, visited, largestValue):
            return None
    return largestValue[0] if largestValue[0] > 0 else None


def test_72():
    # no loop
    nodes = "ABACA"
    edges = [(0, 1), (0, 2), (2, 3), (3, 4)]
    findLargestValuePath(nodes, edges) == 3
    # self loop
    nodes = "A"
    edges = [(0, 0)]
    findLargestValuePath(nodes, edges) is None
    # other loop
    nodes = "ABACA"
    edges = [(0, 1), (0, 2), (2, 3), (3, 2), (3, 4)]
    findLargestValuePath(nodes, edges) is None
