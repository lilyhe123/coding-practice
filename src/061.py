"""
question 61
Implement integer exponentiation. That is, implement the pow(x, y) function,
where x and y are integers and returns x^y.
Do this faster than the naive method of repeated multiplication.
For example, pow(2, 10) should return 1024.
-------------------

2*2...*2
x**y
x**y/2
10: 5+5
   (2+3)
   2*2+1

f(x, y):
  if y == 1: return x
  if y == 2: return x*x
  if y is even: f(x, y/2) * f(x, y/2)
  else: f(x, y/2) * f(x, y/2) * x

f(10): r * r-> f(5): v*v*x -> f(2):v
time O(lgy), space O(lgy)
"""


def pow(x, y):
    if y == 1:
        return x
    if y == 2:
        return x * x

    if y % 2 == 0:
        val = pow(x, y // 2)
        return val * val
    else:
        val = pow(x, y // 2)
        return val * val * x


def test_61():
    assert pow(2, 10) == 1024
    assert pow(2, 9) == 512
