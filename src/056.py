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
