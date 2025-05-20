"""
question 47
Given a array of numbers representing the stock prices of a company in
chronological order, write a function that calculates the maximum profit
you could have made from buying and selling that stock once.
You must buy before you can sell it.

For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you
could buy the stock at 5 dollars and sell it at 10 dollars.
-------------------

1. brute-force
check every possible pair and return the max diff. time O(n^2), space O(1)

2. iterate the array and keep tracking smallest number in previous.
time O(n), space O(1)
"""


def getMaxProfit(nums):
    pre_smallest = nums[0]
    diff = 0
    for i in range(1, len(nums)):
        if nums[i] > pre_smallest:
            diff = max(diff, nums[i] - pre_smallest)
        else:
            pre_smallest = nums[i]
    return diff


def test_47():
    print(getMaxProfit([9, 11, 8, 5, 7, 10]))
