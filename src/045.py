"""
question 45
Using a function rand5() that returns an integer from 1 to 5 (inclusive)
with uniform probability,
implement a function rand7() that returns an integer from 1 to 7 (inclusive).

rand5()  [1, 5):  1, 2, 3, 4, 5
rand7() [1, 7):  1, 2, 3, 4, 5, 6, 7

Hints:
- expend the range: how you can use mulitple call of ran5() to generate a
  number in a range larger than [1,5]
- uniform probability is the key. It means each number in the range has an
  equal chance to be returned.
- rejection sampling: If you only want to use  a sub-range of a range,
  when the outcome falls within the sub-range, you can go ahead and use it.
  If it falls outside the sub-range, you can discard it and try generating
  a new outcome.

consider this:
- How can you combine the results of two rand5() calls (let's say the
  results are x and y, both between 1 and 5) to get a number in a range
  larger than 5?
- Once you have the larger range, can you identify a sub-range within it
  that is a multiple of 7 that you can use for your mapping? Think about
  using modulo operator. But be careful about the the starting value
  and the distribution

a
11, 12, 13, 14, 15
21, 22, 23, 24, 25
31, 32, 33, 34, 35
41, 42, 43, 44, 45
51, 52, 53, 54, 55

b
1,  2,  3,  4,  5
6,  7,  8,  9,  10
11, 12, 13, 14, 15
16, 17, 18, 19, 20
21, 22, 23, 24, 25

random [1, 25]
base 5
c = rand5() * 5 + rand5() - 5

# if in the range [1, 21], module with 7
if c <= 21
  return c % 7 + 1
else: try again
random [1, 7]
"""

import random

# def rand7Again():
#     # generate uniform random number in [0, 24]
#     sum = rand5() * 5 + rand5() - 6
#     # [0, 6], [7, 13], [14, 20]
#     if sum < 21:
#         #
#         return sum % 7 + 1
#     else:
#         return rand7Again()


def rand7():
    def rand5():
        return random.randint(1, 5)

    # generate uniform random number in [1, 25]
    sum = rand5() * 5 + rand5() - 5
    if sum < 22:
        #
        return sum % 7 + 1
    else:
        return rand7()


def test_45():
    total = 1000000
    map = {}
    for i in range(total):
        val = rand7()
        if val not in map:
            map[val] = 0
        map[val] += 1

    alist = list(map.keys())
    alist.sort()
    for num in alist:
        print(num, map[num] / total * 100)
