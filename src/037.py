"""
question 37
The power set of a set is the set of all its subsets. Write a function that,
given a set, generates its power set.

For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3},
{1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.

You may also use a list or array to represent a set.
-------------------

f(set, start):
  f(set,start+1) | {set[start], f(set, start + 1)}
"""


def getPowerSet(alist):
    def getSubset(alist, start):
        if start == len(alist) - 1:
            return [[alist[start]]]
        subList = getSubset(alist, start + 1)
        newList = [[alist[start]]]
        newList.extend(subList)
        for item in subList:
            newItem = [alist[start]]
            for one in item:
                newItem.append(one)
            newList.append(newItem)
        return newList

    return [] + getSubset(alist, 0)


def test_37():
    print(getPowerSet([1, 2, 3]))


def test_38():
    pass
