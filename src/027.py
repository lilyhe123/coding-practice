"""
question 27
Given a string of round, curly, and square open and closing brackets, return
whether the brackets are balanced (well-formed).

For example, given the string "([])[]({})", you should return true.

Given the string "([)]" or "((()", you should return false.
-------------------

Instintively, a stack can help use here but why?
To do this, iterate through the string:
For every opening, it needs to pair to a later unmatched closing. But considering
nesting scenarios, an opening can follow up another opening. So we need to reserve
unmatched openings somewhere. when a closing is comming, it needs to pair with
the most recent unmatched opening.
So a stack, as its Last-In-First-Out nature, is a good fit for this
Iterate through the string, process every bracket:
- if it's opening, push to the stack.
- if it's closing, pop from the stack (the stack should be non-empty) and pair them.
  - If not pair, return early.
- Until we complete all brackets in the string. If the stack is empty, it's balanced.
Otherwise, unbalanced
"""


def test_27():
    pass
