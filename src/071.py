"""
question 71
Using a function rand7() that returns an iterger form 1 to 7 (inclusive)
with uniform probability,
implement a function rand5() that returns an integer from 1to 5 (inclusive).
"""

import random


def rand5():
    def rand7():
        return random.randint(1, 7)

    # generate a number in range [1, 49]
    sum = rand7() * 7 + rand7() - 7

    # [1, 45] is the multiple range of [1, 5]. If sum is in range [1, 45],
    # use it to map to [1, 5] via modulo operator. If not, try again
    if sum < 46:
        return sum % 5 + 1
    else:
        return rand5()


def test_71():
    total = 5000000
    map = {}
    for i in range(total):
        val = rand5()
        if val not in map:
            map[val] = 0
        map[val] += 1

    alist = list(map.keys())
    alist.sort()
    for num in alist:
        print(num, map[num] / total * 100)
