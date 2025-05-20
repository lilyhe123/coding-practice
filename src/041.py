"""
question 41
Given an unordered list of flights taken by someone, each represented
as (origin, destination) pairs, and a starting airport, compute the person's
itinerary. If no such itinerary exists, return null.
If there are multiple possible itineraries, return the lexicographically
smallest one.
All flights must be used in the itinerary.

For example, given the list of flights
[('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')]
and starting airport 'YUL',
you should return the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].

Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting
 airport 'COM', you should return null.

Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
and starting airport 'A',
 you should return the list ['A', 'B', 'C', 'A', 'C'] even though
 ['A', 'C', 'A', 'B', 'C'] is also a valid
 itinerary. However, the first one is lexicographically smaller.
-------------------

A: [B, C]
B: [C]
C: [A]
{'A' -> [['B', False], ['C', False]], 'B' -> [['C', False]], 'C' -> [['A', False]]} # noqa: E501

f(adjList, startAirport, remainingFlightsNum, list):
  if remainingFlightsNum == 0:
    return list
  for nextAirport, visited in enumerate(adjlist[startAirport]):
    if visited: continue

    rtn = f(adjList, nextAirport, remainingFlightsNum-1, list.add(nextAirport)): # noqa: E501
    if rtn: return rtn
  return None

use tuple as the hashkey in a visited map
"""


def smallestItinerary(flights: list[list[str]], start: str) -> list[str]:
    # construct the graph as adjList
    amap = {}
    visited = {}  # (src, dest) -> 0
    for flight in flights:
        src, dest = flight
        visited[(src, dest)] = 0
        if src not in amap:
            amap[src] = []
        amap[src].append(dest)
    for _, value in amap.items():
        value.sort()

    def dfs(amap, visited, start, rtnList) -> list[str]:
        if len(rtnList) == len(visited) + 1:
            return rtnList

        for next in amap[start]:
            if not visited[(start, next)]:
                visited[(start, next)] -= 1
                rtnList.append(next)
                rtn = dfs(amap, visited, next, rtnList)
                if rtn:
                    return rtn
                rtnList.pop(-1)
                visited[(start, next)] += 1
        return None

    return dfs(amap, visited, start, [start])


def test_41():
    print(smallestItinerary([["A", "B"], ["A", "C"], ["B", "C"], ["C", "A"]], "A"))
