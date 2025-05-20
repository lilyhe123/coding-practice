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
