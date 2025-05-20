"""
question 85
Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0,
using only mathematical or bit operations. You can assume b can only be 1 or 0.

-------------------

"""


def x_or_y(x, y, b):
    return x * b + y * (1 - b)


def test_85():
    assert x_or_y(10, 12, 1) == 10
    assert x_or_y(10, 12, 0) == 12


test_85()
