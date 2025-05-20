"""
question 42
Given a list of integers S and a target number k, write a function that
returns a subset of S that adds up to k. If such a subset cannot be made,
then return null.

Integers can appear more than once in the list. You may assume all numbers
in the list are positive.

For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1]
since it sums up to 24.
-------------------

[12, 1, 61, 5, 9, 2]
  ^

return a set of possible sums that is not greater than k
f(aset, start):
 subset = f(aset, sart+1)
 newset = set(subset)
 for arr in subset:
   if sum(arr) + k <= k:
     newset.add
recursion

f([12, 1, 61, 5, 9, 2], 24)
              ^
  f(0, 24, []) or f(0, 12, [12])
    f(1, 24, []) f(1, 23, [1]), f(1, 12, [12]), f(1, 11, [12, 1])
      f(2, 24, []),  f(2, 23, [1]), f(2, 12, [12]), f(2, 11, [12, 1])

top-down:
f(n, target, list)
  if target == 0: return list
  if target < 0: return None
  rtn = f(n+1, target, list)
  if rtn: return rtn
  rtn = f(n+1, target - n, list + nums[n])
  return rtn

bottom-up: (time O(n^2), space O(n^2))
dp array is subset list ended in ith element
for dp[i], iterate dp[0] to dp[i-1], for every subset, add a new item
if total sum < K, if sum == k, return
[12, 1, 61, 5, 9, 2]
               ^
{[12]}, {[12,1], [1]}, None, {[12, 5], [12,1,5], {1, 5}}, {[12,], []}
"""


def test_42():
    pass
