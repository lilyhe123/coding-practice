"""
question 40
Given an array of integers where every integer occurs three times except
for one integer, which only occurs once, find and return the
non-duplicated integer.
For example, given [6, 1, 3, 3, 3, 6, 6], return 1.
Given [13, 19, 13, 13], return 19.
Do this in O(N) time and O(1) space.
-------------------

input: an integer array, every integer occures 3 times except for one
integer, which only occurs once
output: return the non-duplicated integer.
[6, 1, 3, 3, 3, 6, 6]
 ^  ^
 use bit manipulation. XOR etc
"""
