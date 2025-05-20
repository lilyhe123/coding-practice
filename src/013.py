"""
question 13
Given an integer k and a string s, find the length of the longest substring
 that contains at most k distinct characters.

For example, given s = "abcba" and k = 2, the longest substring with k
distinct characters is "bcb".
-------------------

abcba
   i
    j
map: char->freq
if len(map) <= k: j++
else:
  find a local longest substr[i,j)
  move i forward until len(map) == k
"""


def longestSub(s, k):
    i = 0
    freqMap = {}
    maxLen = 1
    for j in range(0, len(s)):
        if s[j] not in freqMap:
            freqMap[s[j]] = 0
        freqMap[s[j]] += 1
        if len(freqMap) > k:
            maxLen = max(maxLen, j - i)
            while len(freqMap) > k:
                freqMap[s[i]] -= 1
                if freqMap[s[i]] == 0:
                    freqMap.pop(s[i])  # remove a key from map
                i += 1
    maxLen = max(maxLen, len(s) - i)
    return maxLen


def test_13():
    print(longestSub("abcba", 2))
