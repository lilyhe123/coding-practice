"""
question 66
Assume you have access to a function toss_biased() which returns 0 or 1 with
a probability that's not 50-50 (but also not 0-100 or 100-0).
You do not know the bias of the coin.

Write a function to simulate an unbiased coin toss.
-------------------

0:1: 3:7
0:1: 7:3
---------
0:1: 10:10
switch 0 and 1 every second time
"""

import random


def toss_unbiased():
    def toss_biased():
        if random.randint(1, 10) <= 3:
            return 0
        else:
            return 1

    v1 = toss_biased()
    v2 = toss_biased()
    if v1 ^ v2:
        return v1
    else:
        return toss_unbiased()


def test_66():
    print("run test66")
    total = 99999
    c1, c2 = 0, 0
    nums = []
    for i in range(total):
        nums.append(toss_unbiased())
        if nums[-1]:
            c1 += 1
        else:
            c2 += 1
    diff = c1 / total * 100 - c2 / total * 100
    print(diff)
    assert -1 < diff < 1
