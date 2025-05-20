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
