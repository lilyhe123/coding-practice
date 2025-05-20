"""
question 12
There exists a staircase with N steps, and you can climb up either 1 or 2
steps at a time. Given N, write a function that returns the number of unique
ways you can climb the staircase. The order of the steps matters.

For example, if N is 4, then there are 5 unique ways:
-------------------

1, 1, 1, 1
2, 1, 1
1, 2, 1
1, 1, 2
2, 2
What if, instead of being able to climb 1 or 2 steps at a time, you could
climb any number from a set of positive integers X? For example,
if X = {1, 3, 5}, you could climb 1, 3, or 5 steps at a time.

input: N steps of a stair case and a set for possible steps for each step.
output: the number of unique ways to clib the stari case.

N steps
X = {s1, s2...sm}, size of M
first let's try to find a way to break down th problem into smaller problem.

f(N, X) = f(N-s1, X) + f(N-s2, X)... + f(N-sm, X)

bottom-up
build up the solution from the base cases to the final value of N

1, 3, 5
  0 1 2 3 4 5 6 7 8
0 1 1 1 2 3

Notes:
- If order matters, dp is an array
  for dp[i], try to apply all the steps {s1,s2...sm}.
  formular dp[i] = dp[i-s1] + dp[i-s2] +...+dp[i-sm]
- If order doen't matter, it means different order with the same numbers
  only count once so only apply the steps one by one in sequence
  dp is a matrix, apply the {s1,s2...sm} one by one
"""


def totalWaysToClimb(n, steps):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(n + 1):
        for step in steps:
            if i >= step:
                dp[i] += dp[i - step]

    return dp[n]


def test_12():
    assert totalWaysToClimb(4, [1, 2]) == 5
    assert totalWaysToClimb(10, [1, 3, 5]) == 47
