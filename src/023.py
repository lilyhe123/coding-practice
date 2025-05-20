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
-------------------

From starting point, try every possible paths until reaching the end point.
Both DFS and BFS work fine here.
But to find the minimum steps, we need to use BFS.
"""

from collections import deque


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
