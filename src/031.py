"""
question 31
The edit distance between two strings refers to the minimum number of character
insertions, deletions, and substitutions required to change one string to the
other. For example, the edit distance between “kitten” and “sitting” is three:
substitute the “k” for “s”, substitute the “e” for “i”, and append a “g”.

Given two strings, compute the edit distance between them.

-------------------
kitten
^
sitting
^

 if c1==c2:
    i1++ i2++
 else:
    a. delete: i1++ -> go right
    b. insert: i2++ -> go down
    c. replace: i1++ i2++ -> go right-down
    steps ++

DP bottom-up
  k i t t e n
s 1 2 3 4 5 6
i 2 1 2 3 4 5
t 3 2 1 2 3 4
t 4 3 2 1 2 3
i 5 4 3 2 2 3
n 6 5 4 3 3 2
g 7 6 5 4 4 3
"""


def editDistance(src, dest):
    init = 0 if src[0] == dest[0] else 1
    pre = list(range(init, init + len(src)))
    cur = [0] * len(src)
    for i in range(1, len(dest)):
        cur[0] = pre[0] + 1
        for j in range(1, len(src)):
            if src[j] == dest[i]:
                cur[j] = pre[j - 1]
            else:
                cur[j] = min(min(pre[j - 1], pre[j]), cur[j - 1]) + 1
        pre = cur
        cur = [0] * len(src)
    return pre[len(src) - 1]


def test_31():
    assert editDistance("kitten", "sitting") == 3
    assert editDistance("sigtten", "sitting") == 3
