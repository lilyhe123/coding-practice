"""
question 5
cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the
first and last element of that pair. For example, car(cons(3, 4)) returns 3,
and cdr(cons(3, 4)) returns 4.

Given this implementation of cons:

def cons(a, b):
    def pair(f):
        return f(a, b)
    return pair
Implement car and cdr.

-------------------
Understand the problem
cons() return a function 'pair and 'pair' takes another function 'f' as input.
So car() and cdr() need to provide its own impl of function 'f' and pass it to 'pair'.
"""


def cons(a, b):
    def pair(f):
        return f(a, b)

    return pair


def car(pair):
    def getFirst(a, b):
        return a

    return pair(getFirst)


def cdr(pair):
    def getSecond(a, b):
        return b

    return pair(getSecond)


def test_5():
    assert car(cons(3, 4)) == 3
    assert cdr(cons(3, 4)) == 4
