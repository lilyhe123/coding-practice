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
