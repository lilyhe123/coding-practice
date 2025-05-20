"""
question 15 TODO
Given a stream of elements too large to store in memory, pick a random
element from the stream with uniform probability.
-------------------

break down the problem
let's assume already pick an element randomly in [1, i].
Now i+1 element comes in, we need to pick up the i+1 element at 1/(i+1)
chance. so if random(1, i+1) == i+1, change to picked elment to t
he i+1 element. otherwise, the picked element remains no change.

reservoir sampling
if the stream totally have 5 elements:
probability to choose the 1th element: 1 * 1/2 * 2/3 * 3/4 * 4/5 = 1/5
probability to choose the 2th element:     1/2 * 2/3 * 3/4 * 4/5 = 1/5
probability to choose the 3th element:           1/3 * 3/4 * 4/5 = 1/5
probability to choose the 4th element:                 1/4 * 4/5 = 1/5
probability to choose the 5th element:                       1/5 = 1/5
"""

import random

import matplotlib.pyplot as plt
import numpy as np


def getRandomFromStream(stream):
    rst = 0
    for i, val in enumerate(stream, 1):
        if random.randint(1, i + 1) == i:
            rst = val

    return rst


def create_hist(stream, no_of_samples=5000):
    hist = []
    for i in range(no_of_samples):
        val = getRandomFromStream(stream)
        hist.append(val)
    plt.hist(np.array(hist), align="left")
    plt.ylabel("Probability")
    plt.show()


def test_15():
    random.seed(1)
    # stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # create_hist(stream, 10000)
