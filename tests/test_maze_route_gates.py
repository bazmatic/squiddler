"""Routes respect the gates as they swing, whatever they blocked a moment ago."""
from types import SimpleNamespace

from hunter.maze_route import MazeRoutes
from maze.config import MazeComplexity
from maze.generator import Maze


def gate(start, end):
    return SimpleNamespace(start=start, end=end, active=True)


def test_a_gate_across_the_way_blocks_the_route_until_it_swings_clear():
    maze = Maze(1, MazeComplexity.EMPTY, 10)
    centre = maze.position_calculator.grid_center_to_screen
    here, there = centre(2, 5), centre(3, 5)
    between = (here[0] + maze.cell_size_x / 2, here[1])
    top, bottom = maze.offset_y - 50, maze.offset_y + maze.grid_height * maze.cell_size_y + 50
    across = gate((between[0], top), (between[0], bottom))          # a plank the height of the maze: no way round it
    routes = MazeRoutes()
    maze.gate_segments = [across]
    assert routes.reachable_targets(maze, here, [there]) == []
    across.start, across.end = (between[0] + 200, top), (between[0] + 200, bottom)     # swung clear of the link
    assert routes.reachable_targets(maze, here, [there]) == [0]
    across.start, across.end = (between[0], top), (between[0], bottom)
    assert routes.reachable_targets(maze, here, [there]) == []
