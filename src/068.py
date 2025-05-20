"""
question 68
On our special chessboard, two bishops attack each other if they share the same
diagonal. This includes bishops that have another bishop located between them,
i.e. bishops can attack through pieces.

You are given N bishops, represented as (row, column) tuples on a M by M
chessboard. Write a function to count the number of pairs of bishops that attack
each other. The ordering of the pair doesn't matter: (1, 2) is considered the
same as (2, 1).

For example, given M = 5 and the list of bishops:

(0, 0)
(1, 2)
(2, 2)
(4, 0)

The board would look like this:

[b 0 0 0 0]
[0 0 b 0 0]
[0 0 b 0 0]
[0 0 0 0 0]
[b 0 0 0 0]

You should return 2, since bishops 1 and 3 attack each other, as well as bishops
3 and 4.

-------------------
1. brute-force approach
Scan every diagonal to count number of bishops. Two categories of diagonals: go
left-down and go right-down.
If one diagonal has k bishops and they can make k(k-1)/2 pairs.
time O(M^2), space O(1)

2. Observed that cells in the same diagonal folow some patterns:
either sums of row index and column index are the same, or diff of row index
and column index are the same.
Create two maps: sum_map: sum->freq and diff_map: dif->freq
So scan the list of bishops, calculate the diff and sum of i and j and populate
the two maps.
time O(N), space O(N)

"""


def get_pairs_of_bishops(bishops: list) -> int:
    sum_map, diff_map = {}, {}
    for i, j in bishops:
        s = i + j
        diff = i - j
        if s not in sum_map:
            sum_map[s] = 0
        sum_map[s] += 1
        if diff not in diff_map:
            diff_map[diff] = 0
        diff_map[diff] += 1
    total = 0
    for freq in sum_map.values():
        if freq > 1:
            total += freq * (freq - 1) // 2
    for freq in diff_map.values():
        if freq > 1:
            total += freq * (freq - 1) // 2
    return total


def test_68():
    bitshops = [(0, 0), (1, 2), (2, 2), (4, 0)]
    assert get_pairs_of_bishops(bitshops) == 2
    bitshops = [(0, 0), (1, 2), (2, 2), (3, 3), (2, 3), (4, 0)]
    assert get_pairs_of_bishops(bitshops) == 5
