"""
question 70
A number is considered perfect if its digits sum up to exactly 10.

Given a positive integer n, return the n-th perfect number.

For example, given 1, you should return 19. Given 2, you should return 28.

-------------------
It's a math problem.
Identify the pattern of the number: 9K+1 with exceptions, e.g. 1, 100, 1000...

9k+1:   10  19 28  37  46  55  64  73  82   91  100  109  118...   991 1000
perfect: n   y  y   y   y   y   y   y   y    y    n   y     y       n     n

Use a counter, iterate all 9K+1, starting from 19,
calculating sum of the number's digits, if sum is 10, increase counter
until counter reaches n, return the number
"""


def sumDigits(num):
    sum = 0
    while num > 0:
        sum += num % 10
        num //= 10
    return sum


def getPerfectNumber(n):
    count = 1
    num = 19
    while count < n:
        num += 9
        if sumDigits(num) == 10:
            count += 1

    return num


def test_70():
    nums = []
    for i in range(1, 500, 50):
        nums.append(getPerfectNumber(i))
    assert nums == [19, 613, 1432, 2422, 4015, 6022, 10180, 11143, 12232, 14041]
