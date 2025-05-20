"""
question 28
Write an algorithm to justify text. Given a sequence of words and an integer
line length k, return a list of strings which represents each line, fully
justified.

More specifically, you should have as many words as possible in each line. There
should be at least one space between each word. Pad extra spaces when necessary
so that each line has exactly length k. Spaces should be distributed as equally
as possible, with the extra spaces, if any, distributed starting from the left.

If you can only fit one word on a line, then you should pad the right-hand side
with spaces.

Each word is guaranteed not to be longer than k.

For example, given the list of words ["the", "quick", "brown", "fox", "jumps",
"over", "the", "lazy", "dog"] and k = 16, you should return the following:

["the  quick brown", # 1 extra space on the left
 "fox  jumps  over", # 2 extra spaces distributed evenly
 "the   lazy   dog"] # 4 extra spaces distributed evenly"

-------------------

iterate words, trying to fit in as much words as possible.
use two pointers, first pointer is the first word of current line and
second pointer is the word to check whether it can be fit in to current line.

["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
   i1
           i2
initially i1 = 0
cur_len = len(word[i1])
when add new word[i2], new length is cur_len + len(word[i2]) + 1
if cur_len is smaller or equal to k:
    move i2 forward.
otherwise:
    create a new line with words from i1 to i2-1;
        distribute the extra spaces evenly between words starting from left
        remaining_spaces = k - cur_len
        number of spaces in words: total_num = i2 - i1
        even space number: even_spaces = 1 + remaining_spaces // total_num
        extra_remaining: remaining_spaces % total_num
    move i1 to i2 as the first word  and move i2 forward
"""


def justify_text(words, k):
    # generate line with words [start, end)
    # cur_len is the length with words and one space between words
    def generate_one_line(start: int, end: int, cur_len: int):
        new_line = ""
        space_num = end - start - 1
        if space_num == 1:
            # only one word in the line
            new_line += words[start]
            new_line += " " * (k - len(words[start]))
            return new_line
        small_space_num = 1 + (k - cur_len) // (space_num)
        small_spaces = " " * small_space_num
        large_spaces = small_spaces + " "
        # print(f"'{large_spaces}'")
        large_space_num = (k - cur_len) % (space_num)
        new_line += words[i1]
        for i in range(1, space_num + 1):
            if i <= large_space_num:
                new_line += large_spaces
            else:
                new_line += small_spaces
            new_line += words[start + i]
        return new_line

    lines = []
    i1 = 0
    cur_len = len(words[0])
    for i2 in range(1, len(words)):
        if cur_len + len(words[i2]) + 1 > k:
            # needs to generate new line with words in [i1, i2)
            lines.append(generate_one_line(i1, i2, cur_len))
            i1 = i2
            cur_len = len(words[i1])
        else:
            cur_len = cur_len + len(words[i2]) + 1
    # process left_over words from i1 to the end
    lines.append(generate_one_line(i1, len(words), cur_len))
    return lines


def test_28():
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    k = 16
    assert justify_text(words, k) == [
        "the  quick brown",
        "fox  jumps  over",
        "the   lazy   dog",
    ]
