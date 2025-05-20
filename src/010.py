"""
question 10
Implement a job scheduler which takes in a function f and an integer n,
and calls f after n milliseconds.
-------------------

!! how to create and run a thread
"""

import time
from threading import Thread


def scheduler():
    def schedule(f, n):
        time.sleep(n / 1000)
        f()

    def hello(name):
        print("hello " + name)

    job = Thread(target=schedule, args=(lambda: hello("from thread"), 200))
    job.start()

    print("Thread, you there?")
    time.sleep(1)
    print("Hello to you too")


def test_10():
    scheduler()
