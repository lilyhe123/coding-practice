"""
question 74
Suppose you have a multiplication table that is N by N. That is, a 2D array
where the value at the i-th row and j-th column is (i + 1) * (j + 1) (if
0-indexed) or i * j (if 1-indexed).

Given integers N and X, write a function that returns the number of times X
appears as a value in an N by N multiplication table.

For example, given N = 6 and X = 12, you should return 4, since the
multiplication table looks like this:

| 1 | 2 | 3 | 4 | 5 | 6 |
| 2 | 4 | 6 | 8 | 10 | 12 |
| 3 | 6 | 9 | 12 | 15 | 18 |
| 4 | 8 | 12 | 16 | 20 | 24 |
| 5 | 10 | 15 | 20 | 25 | 30 |
| 6 | 12 | 18 | 24 | 30 | 36 |

And there are 4 12's in the table.

-------------------
1. brute force: check every cell in the N*N table. count the occurrance of X.
   time O(N*N), space O(1)

2. basically we are looking for pairs of integers (i, j)
   that i*j = X and both x and j in range [1, N]
   iterate integer in range [1, N], if the integer is divided by X and the result of the
   division is also in range [1, N], there is one occurence of X.
   time O(N), space O(1)

edge cases: X < N
"""


def getOccurance(N: int, X: int):
    end = min(N, X)
    count = 0
    for i in range(1, end + 1):
        if X % i == 0 and X // i <= end:
            count += 1
    return count


def test_74():
    assert getOccurance(6, 12) == 4
    assert getOccurance(6, 20) == 2
    assert getOccurance(6, 2) == 2
