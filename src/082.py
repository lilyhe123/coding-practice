"""
question 82
Using a read7() method that returns 7 characters from a file, implement readN(n)
which reads n characters.

For example, given a file with the content “Hello world”, three read7() returns
“Hello w”, “orld” and then “”.

-------------------

"""

input = "abcdefghijklmnopqrstuvw"
idx = 0


def read7():
    global idx
    diff = min(7, len(input) - idx)
    idx += diff
    return input[idx - diff : idx]


def readN(n):
    li = []
    while n > 0:
        s = read7()
        if s == "":
            break
        elif n < len(s):
            li.append(s[:n])
        else:
            li.append(s)
        n -= 7
    return "".join(li)


def test_82():
    global idx
    idx = 0
    assert len(readN(20)) == 20
    idx = 0
    assert len(readN(14)) == 14
    idx = 0
    assert len(readN(50)) == len(input)
