import pytest
from hunter.visibility import visible_point, visible_wall_portions


def test_wall_hides_contact_and_range_limits_visibility():
    walls = [((5.,-2.),(5.,2.))]
    assert visible_point((0,0),(4,0),walls,10)
    assert visible_point((0,0),(5,0),walls,10)
    assert not visible_point((0,0),(8,0),walls,10)
    assert not visible_point((0,0),(0,11),walls,10)
    assert not visible_point((0,0),(8,0), [((3,0),(5,0))],10)


def test_wall_portions_clip_range_and_hidden_ends():
    front = ((5.,-2.),(5.,2.))
    back = ((8.,-20.),(8.,20.))
    pieces = visible_wall_portions((0,0), [front, front, back], 10)
    assert len([p for p in pieces if p[0][0] == 5]) == 1
    for a,b in pieces:
        assert a[0]**2+a[1]**2 <= 100.00001
        assert b[0]**2+b[1]**2 <= 100.00001
        if a[0] == 8:
            assert not (-3.19 < (a[1]+b[1])/2 < 3.19)


# --- the fast implementation must agree with the original ---

import random
import time

from hunter.visibility import local_segments
from tests import visibility_reference as reference


def random_maze_walls(rng, cells=12, size=50.0, fill=0.35):
    """Cell-edge wall segments like the maze converter emits, duplicates and all."""
    walls = []
    for gx in range(cells):
        for gy in range(cells):
            if rng.random() < fill:
                x, y = gx * size, gy * size
                walls += [((x, y), (x + size, y)), ((x + size, y), (x + size, y + size)),
                          ((x + size, y + size), (x, y + size)), ((x, y + size), (x, y))]
    return walls


def canonical(pieces):
    return sorted(tuple(sorted((round(a[0], 5), round(a[1], 5)) for a in piece)) for piece in pieces)


@pytest.mark.parametrize('seed', range(30))
def test_fast_visibility_matches_the_original(seed):
    rng = random.Random(seed)
    walls = random_maze_walls(rng)
    if seed % 3 == 0:  # Some walls at odd angles too
        walls += [((rng.uniform(0, 600), rng.uniform(0, 600)), (rng.uniform(0, 600), rng.uniform(0, 600)))
                  for _ in range(6)]
    for _ in range(6):
        origin = (rng.uniform(0, 600), rng.uniform(0, 600))
        radius = rng.choice((80.0, 200.0, 400.0))
        assert canonical(local_segments(origin, walls, radius)) == canonical(
            reference.local_segments(origin, walls, radius))
        assert canonical(visible_wall_portions(origin, walls, radius)) == canonical(
            reference.visible_wall_portions(origin, walls, radius))
        local = reference.local_segments(origin, walls, radius)
        for _ in range(40):
            point = (origin[0] + rng.uniform(-radius, radius), origin[1] + rng.uniform(-radius, radius))
            assert visible_point(origin, point, local, radius) == reference.visible_point(origin, point, local, radius)


def test_fast_visibility_matches_on_grid_aligned_origins():
    """Origins on cell centres and corners line rays up with wall ends: the awkward cases."""
    walls = random_maze_walls(random.Random(99))
    for origin in ((275.0, 275.0), (300.0, 300.0), (250.0, 325.0), (300.0, 275.0)):
        assert canonical(visible_wall_portions(origin, walls, 400.0)) == canonical(
            reference.visible_wall_portions(origin, walls, 400.0))


@pytest.mark.perf
def test_fast_visibility_is_much_faster_in_a_dense_maze():
    walls = random_maze_walls(random.Random(7), cells=30, size=33.0)
    origin = (495.0 + 16.5, 495.0 + 16.5)

    def timed(function):
        best = float('inf')
        for _ in range(3):
            start = time.perf_counter()
            function(origin, walls, 400.0)
            best = min(best, time.perf_counter() - start)
        return best

    assert timed(visible_wall_portions) * 4 < timed(reference.visible_wall_portions)


@pytest.mark.perf
def test_visibility_on_curved_maze_stays_fast():
    """Hunter sight on curved rock, with its many short walls, costs at most 4x what square walls cost."""
    import pygame
    from maze.config import MazeComplexity
    from maze.generator import Maze

    pygame.init()

    def maze(curved):
        random.seed(125)
        return Maze(1, complexity=MazeComplexity.COMPLEX, grid_size=32, seed=125, curved=curved)

    curved, square = maze(True), maze(False)
    assert curved.grid == square.grid
    opens = [(x, y) for y, row in enumerate(curved.grid) for x, value in enumerate(row) if value == 0]
    origins = [curved.position_calculator.grid_center_to_screen(*cell) for cell in random.Random(5).sample(opens, 20)]

    def timed(m):
        walls = m.sight_segments()  # What hunter perception sees by
        best = float('inf')
        for _ in range(3):
            start = time.perf_counter()
            for origin in origins:
                visible_wall_portions(origin, walls, 400.0)
            best = min(best, time.perf_counter() - start)
        return best

    assert timed(curved) < 4 * timed(square)


def test_sight_outline_stays_close_to_the_rock():
    """Merged sight walls: far fewer on curved rock, every corner they cut off within tolerance,
    and square walls passed through untouched."""
    import math
    import pygame
    from hunter.visibility import SIGHT_TOLERANCE, simplify_outline
    from maze.config import MazeComplexity
    from maze.generator import Maze

    pygame.init()
    random.seed(125)
    curved = Maze(1, complexity=MazeComplexity.COMPLEX, grid_size=32, seed=125)
    raw = [(w.start, w.end) for w in curved.walls if w.active]
    sight = curved.sight_segments()
    assert len(sight) < 0.6 * len(raw)
    assert curved.sight_segments() is sight  # cached until the walls change

    def distance(point, segment):
        (ax, ay), (bx, by) = segment
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / (dx * dx + dy * dy)))
        return math.hypot(point[0] - ax - dx * t, point[1] - ay - dy * t)

    # Every original corner lies within tolerance of some sight wall
    for start, end in raw:
        for point in (start, end):
            assert min(distance(point, wall) for wall in sight) <= SIGHT_TOLERANCE + 1e-6

    random.seed(125)
    square = Maze(1, complexity=MazeComplexity.COMPLEX, grid_size=32, seed=125, curved=False)
    assert square.sight_segments() == [(w.start, w.end) for w in square.walls if w.active]
    walls = random_maze_walls(random.Random(3))
    assert sorted(simplify_outline(walls)) == sorted(walls)
