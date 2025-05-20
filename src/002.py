"""
!! question 2
Given an array of integers, return a new array such that each element at index
 i of the new array is the product of all the numbers in the original array
 except the one at i.

For example, if our input was [1, 2, 3, 4, 5],
the expected output would be [120, 60, 40, 30, 24].
If our input was [3, 2, 1], the expected output would be [2, 3, 6].

Follow-up: what if you can't use division?
-------------------

## brainstorming
1. brute-force: for every number, calculate the product of all the
   other number.
   time O(n^2), space O(1)
2. in the brute force, we do many duplcated calculation:
   the product of the same numbers.
   if we can prevent the duplicated calculation we can optimize it.
   We can first calculate the product of all the numbers in the array.
   for every element, we only need to use the current nuber to divide
   the total product and get the result we need.
   time O(n), space O(1)
3. what if we can't use division?
  brute-force approach doesn't use division, but can we optimize it?

  formula for #2 is totalProduct/currentNumber
  new formula: product of numbers in its left * product of numbers
  in its right
  prefix and suffix products
  so we need two arrays, the first is the product of left numbers
  and the second is the product of right numbers.
  left_prod[i] = arr[0]*arr[1]..*arr[i-1]
  right_prod[i] = arr[i+1]* arr[i+2]...arr[n-1]
  output[i] = left_prod[i] * right_prod[i]

  !! optimze space - in-place calculation
  do the calculation directly in the result
  result[0] = 1
  i from 1 to n-1:
    result[i] = result[i-1] * nums[i-1]
  prod = 1
  i form n-2 to 0:
    prod *= nums[i+1]
    result[i] *= prod

   time O(n), space O(n)

  edge case: if there is a zero in the input array, the approach without
  division works correctly without specific handling this case.
"""


def test_2():
    pass
