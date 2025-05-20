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
