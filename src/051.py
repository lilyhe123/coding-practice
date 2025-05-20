import random

# uncomments the following line when need to debug stack overflow error
# import sys
# sys.setrecursionlimit(10)
"""
question 51
Given a function that generates perfectly random numbers between 1 and k
(inclusive), where k is an input, write a function that shuffles a deck
of cards represented as an array using only swaps.

It should run in O(N) time.

Hint: Make sure each one of the 52! permutations of the deck is
equally likely.
-------------------

input: random(k) return a peerfectrly random number between 1 and k,
k is an input.
contraints: using only swaps?
every one selection from i cards(i=52,51...1) should have equal probability
rand(52)*rand(51)
[1,52]
52, random[1,51];

"""


def shuffleCards():
    def randomGen(k):
        return random.randint(1, k)

    def swap(cards, i, j):
        tmp = cards[i]
        cards[i] = cards[j]
        cards[j] = tmp

    cards = [i for i in range(52)]
    for i in range(51, 0, -1):
        j = randomGen(i) - 1
        swap(cards, i, j)
    return cards


def test_51():
    print(shuffleCards())
    print(shuffleCards())
