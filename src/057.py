"""
question 57
Given a string s and an integer k, break up the string into multiple lines
such that each line has a length of k or less. You must break it up so that
words don't break across lines. Each line has to have the maximum possible
amount of words. If there's no way to break the text up, then return null.

You can assume that there are no spaces at the ends of the string and that
there is exactly one space between each word.

For example, given the string "the quick brown fox jumps over the lazy dog"
and k = 10, you should return:
["the quick", "brown fox", "jumps over", "the lazy", "dog"].
No string in the list has a length of more than 10.
-------------------

k = 10
"the quick brown fox jumps over the lazy dog"

tokens: ['the', 'quick', 'brown', 'fox']

time complexity:
- splitting the string into words, O(n), n is the length of the string.
- iterate words, O(m), m is the number of words
So overall time complexity is O(n+m).

space complexity:
We create data structure to store words and lines.
So space complexity is O(m).

"""


def break_str(s, k):
    tokens = s.split(" ")
    maxLen = max(len(x) for x in tokens)
    if maxLen > k:
        return None
    # line is always not empty
    line = tokens[0]
    line_list = []
    for i in range(1, len(tokens)):
        remainingLen = k - len(line) - 1
        if len(tokens[i]) <= remainingLen:
            line += " "
            line += tokens[i]
        else:
            line_list.append(line)
            line = tokens[i]
    # handle leftover
    line_list.append(line)
    return line_list


def test_57():
    s = "the quick brown fox jumps over the lazy dog"
    k = 10
    print(break_str(s, k))
    k = 4
    print(break_str(s, k))
