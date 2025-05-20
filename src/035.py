"""
question 35
Given an array of strictly the characters 'R', 'G', and 'B', segregate the
values of the array so that all the Rs come first, the Gs come second, and t
he Bs come last. You can only swap elements of the array.

Do this in linear time and in-place.

For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should
become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].
-------------------

p1 is the for the next R,
p3 is for the next B,
p2 is the next value to check
['R', 'R', 'R', 'G', 'G', 'B', 'B']
            1         2
                           3
"""


def sortRGBArray(arr):
    def swap(arr, i, j):
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp

    p1, p2, p3 = 0, 0, len(arr) - 1

    while p2 <= p3:
        if arr[p2] == "R":
            swap(arr, p1, p2)
            p1 += 1
            p2 += 1
        elif arr[p2] == "B":
            swap(arr, p2, p3)
            p3 -= 1
        else:
            p2 += 1


def test_35():
    arr = ["G", "B", "R", "R", "B", "R", "G"]
    sortRGBArray(arr)
    print(arr)
