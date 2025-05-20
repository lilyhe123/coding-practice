"""
question 81
Given a mapping of digits to letters (as in a phone number), and a digit string,
return all possible letters the number could represent. You can assume each
valid number in the mapping is a single digit.

For example if {“2”: [“a”, “b”, “c”], 3: [“d”, “e”, “f”], …} then “23” should
return [“ad”, “ae”, “af”, “bd”, “be”, “bf”, “cd”, “ce”, “cf"].

-------------------
One digit can be mapped to different letters. We need to exhaust all the combinations
of the possible letters of each digit.

For a two-digit number, we can use two nested loops to exhaust all the combinations of
the two lists.
For a n-digit number, we need more general way to handle it.
There are two approaches: iteration and recursion

Iteration
Create a queue and each element in the queue is a list of string.
Initially for every digit of the given number, get its mapped letters from the map and
store it to the queue.

Then we can iterate the queue until only one element left.
  At each step remove two elements from the queue and merge them into one,
  with all combinations of the two list. Then add the merged list to the queue.
Return the last element.

Recursion
Break down the problem into smaller problems.
The problem is that for a given number we need to return a list of string reprented by letter.
For a n-digit number, split it to two parts: first digit and n-1 digits.
First sesolve the two smaller problels seperately and recursively.
Then merge the returned two lists into one and return.
The base case is when there is only one digit in the number, return its mapped letters directly.

Let's do this in recursive way.
"""


def convertToLetters(map, num):
    def dfs(s):
        if len(s) == 1:
            return map[s]
        rtn = []
        l1 = dfs(s[0])
        l2 = dfs(s[1:])
        for e1 in l1:
            for e2 in l2:
                rtn.append(e1 + e2)
        return rtn

    return dfs(num)


def test_81():
    map = {"2": ["a", "b", "c"], "3": ["d", "e", "f"], "4": ["h", "i"]}
    num = "23"
    assert convertToLetters(map, num) == [
        "ad",
        "ae",
        "af",
        "bd",
        "be",
        "bf",
        "cd",
        "ce",
        "cf",
    ]
    num = "342"
    assert convertToLetters(map, num) == [
        "dha",
        "dhb",
        "dhc",
        "dia",
        "dib",
        "dic",
        "eha",
        "ehb",
        "ehc",
        "eia",
        "eib",
        "eic",
        "fha",
        "fhb",
        "fhc",
        "fia",
        "fib",
        "fic",
    ]
