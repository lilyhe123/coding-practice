"""
question 39
Conway's Game of Life takes place on an infinite two-dimensional board of
square cells. Each cell is either dead or alive, and at each tick,
the following rules apply:

Any live cell with less than two live neighbours dies.
Any live cell with two or three live neighbours remains living.
Any live cell with more than three live neighbours dies.
Any dead cell with exactly three live neighbours becomes a live cell.
A cell neighbours another cell if it is horizontally, vertically,
or diagonally adjacent.

Implement Conway's Game of Life. It should be able to be initialized with
a starting list of live cell coordinates and the number of steps it should
run for. Once initialized, it should print out the board state at each step.
Since it's an infinite board, print out only the relevant coordinates, i.e.
from the top-leftmost live cell to bottom-rightmost live cell.

You can represent a live cell with an asterisk (*) and
a dead cell with a dot (.).
-------------------

neighbours: 8
live neighbours:
 < 2: die
 2, 3: keep living
 > 3: die
 3: die -> living

die -> living
living -> die

 input: a starting list of live cell coordinates,
 the number of steps it should run for?
 live cell set: {(row, col)...}
 add live cells and neighbours of every live cell to a map: cell ->living|die
 iterate cells in the map, apply the rules which might result in changing
 its stat. put live cells to a new set
 auxiliary methods:
   getAllNeighbours(row, col)->List<(int, int)>:
   getLiveNeighbourCount(map, row, col)->int

"""


class GameOfLife:
    def __init__(self, liveCells: list, steps=0):
        self.live_cells = set(tuple(cell) for cell in liveCells)
        self.steps = steps

    def getAllNeighbours(self, cell):
        neighbours = []
        row, col = cell
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if (i, j) != cell:
                    neighbours.append((i, j))
        return neighbours

    def getLiveNeighbourCount(self, cell):
        row, col = cell
        count = 0
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if (i, j) != cell and (i, j) in self.live_cells:
                    count += 1
        return count

    def nextGen(self):
        next_live_cells = set()
        to_check_cells = set()
        for cell in self.live_cells:
            for neighbour in self.getAllNeighbours(cell):
                to_check_cells.add(neighbour)

        for cell in to_check_cells:
            count = self.getLiveNeighbourCount(cell)
            # remain living
            if (count == 2 or count == 3) and cell in self.live_cells:
                next_live_cells.add(cell)
            # from die to living
            if count == 3 and cell not in self.live_cells:
                next_live_cells.add(cell)
        self.live_cells = next_live_cells

    def run(self):
        for _ in range(self.steps):
            print(self.live_cells)
            self.nextGen()


def test_39():
    # Example 1: Blinker pattern
    print("Blinker pattern")
    initial_blinker = [(0, 1), (1, 1), (2, 1)]
    blinker_pattern = GameOfLife(initial_blinker, 5)
    blinker_pattern.run()

    # Example 2: Toad pattern
    print("Toad pattern")
    initial_toad = [(1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)]
    toad_pattern = GameOfLife(initial_toad, 5)
    toad_pattern.run()

    # Example 3: Glider pattern
    print("Glider pattern")
    initial_glider = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    glider_pattern = GameOfLife(initial_glider, 10)
    glider_pattern.run()


def test_40():
    pass
