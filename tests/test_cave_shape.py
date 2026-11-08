"""CaveShape: the cave field's outline as the maze's WallSegments, rebuilt after a destroy."""
import random
import statistics
import time

import numpy as np
import pytest

import config
from maze.cave import CaveShape
from maze.cave_field import THRESHOLD
from maze.config import MazeComplexity, MazeComplexityPresets
from maze.converter import SIDES, GridToWallsConverter
from maze.generator import RecursiveBacktrackingGenerator
from maze.positioning import MazePositionCalculator
from tests import cave_checks


def generated(size=26, seed=125):
    state = random.getstate()
    try:
        random.seed(seed)
        return RecursiveBacktrackingGenerator(
            MazeComplexityPresets.get_config(MazeComplexity.COMPLEX), size, size, 25).generate()
    finally:
        random.setstate(state)


def build(grid, strong=frozenset(), seed=7):
    rows, cols = len(grid), len(grid[0])
    calc = MazePositionCalculator(cols, rows)
    converter = GridToWallsConverter(calc, frozenset(strong))
    shape = CaveShape(grid, calc, converter, strong=frozenset(strong), keep_clear=(), seed=seed,
                      settings=config.CAVE, hit_points=lambda cell: config.WALL_HIT_POINTS)
    return shape, calc, converter


def key(segment):
    return (tuple(round(v, 3) for v in segment.start), tuple(round(v, 3) for v in segment.end), segment.cell)


def interior_exposed_walls(grid):
    rows, cols = len(grid), len(grid[0])
    return [(c, r) for r in range(2, rows-2) for c in range(2, cols-2)
            if grid[r][c] == 1 and any(grid[r+dr][c+dc] == 0 for dc, dr in SIDES)]


@pytest.fixture(scope='module')
def maze():
    grid = generated()
    shape, calc, converter = build(grid)
    return grid, shape, calc, converter


def test_segments_follow_field(maze):
    grid, shape, _, _ = maze
    off = [p for s in shape.segments() for p in (s.start, s.end)
           if abs(shape.field.value_at(*p) - THRESHOLD) >= 0.08]
    assert off == []


def test_no_zero_length_segments(maze):
    _, shape, _, _ = maze
    assert all(s.start != s.end for s in shape.segments())


def test_watertight(maze):
    grid, shape, calc, _ = maze
    assert cave_checks.watertight_violations(grid, calc, shape.segments()) == []


def test_every_segment_owned_by_a_wall_cell(maze):
    grid, shape, _, converter = maze
    for s in shape.segments():
        c, r = s.cell
        assert grid[r][c] == 1
        assert s.destructible == (not converter.on_perimeter(grid, c, r))
        assert s.hit_points == config.WALL_HIT_POINTS


def test_steel_faces_identical():
    grid = generated()
    steel = frozenset(interior_exposed_walls(grid)[:3])
    shape, calc, converter = build(grid, steel)
    for cell in steel:
        expected = sorted((f.start, f.end) for f in
                          (converter.face(grid, cell, side, config.WALL_HIT_POINTS) for side in SIDES
                           if not converter.is_wall(grid, cell[0]+side[0], cell[1]+side[1])))
        got = [s for s in shape.segments() if s.cell == cell]
        assert sorted((s.start, s.end) for s in got) == expected
        assert all(not s.destructible for s in got)
    assert cave_checks.watertight_violations(grid, calc, shape.segments()) == []


def test_open_cell_matches_fresh_build():
    grid = generated()
    shape, _, _ = build(grid)
    cols = len(grid[0])
    cell = interior_exposed_walls(grid)[len(interior_exposed_walls(grid)) // 2]
    before = shape.segments()
    old_values, old_owner = shape.field.values.copy(), shape.field.owner.copy()
    version = shape.version

    grid[cell[1]][cell[0]] = 0
    removed, added = shape.open_cell(cell)
    after = shape.segments()

    fresh, _, _ = build([row[:] for row in grid])
    assert sorted(map(key, after)) == sorted(map(key, fresh.segments()))
    assert shape.version == version + 1
    assert all(s.cell != cell for s in after)

    # The diff is exact: before - removed + added == after, by identity
    after_ids = {id(s) for s in after}
    assert {id(s) for s in before} - {id(s) for s in removed} | {id(s) for s in added} == after_ids
    assert not any(id(s) in after_ids for s in removed)

    # Outside the 3x3 tiles around every tile whose samples changed, the objects are the same
    s = shape.field.samples
    diff = (np.abs(shape.field.values - old_values) > 0) | (shape.field.owner != old_owner)
    ys, xs = np.nonzero(diff)
    changed = {(min(x // s.per_x, s.cols-1) + dc, min(y // s.per_y, s.rows-1) + dr)
               for y, x in zip(ys, xs) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    assert all(abs(c - cell[0]) <= 4 and abs(r - cell[1]) <= 4 for c, r in changed)  # stays local

    def tile(seg):
        fy, fx = s.fractional((seg.start[0]+seg.end[0])/2, (seg.start[1]+seg.end[1])/2)
        return (min(int(fx // s.per_x), cols-1), min(int(fy // s.per_y), s.rows-1))

    far = [seg for seg in before if tile(seg) not in changed]
    assert far and all(id(seg) in after_ids for seg in far)


def test_open_cell_beside_steel_exposes_its_face():
    grid = generated()
    rows, cols = len(grid), len(grid[0])
    steel, cell = next(((c, r), (c+1, r)) for r in range(2, rows-2) for c in range(2, cols-3)
                       if grid[r][c] == 1 and grid[r][c+1] == 1 and grid[r][c+2] == 0)
    shape, calc, converter = build(grid, {steel})
    grid[cell[1]][cell[0]] = 0
    shape.open_cell(cell)
    fresh, _, _ = build([row[:] for row in grid], {steel})
    assert sorted(map(key, shape.segments())) == sorted(map(key, fresh.segments()))
    face = converter.face(grid, steel, (1, 0), config.WALL_HIT_POINTS)
    assert (face.start, face.end) in {(s.start, s.end) for s in shape.segments() if s.cell == steel}
    assert cave_checks.watertight_violations(grid, calc, shape.segments()) == []


def test_rock_and_territory(maze):
    grid, shape, calc, _ = maze
    wall = interior_exposed_walls(grid)[0]
    x, y = calc.grid_to_screen(*wall)
    assert shape.is_rock((x + calc.cell_size_x/2, y + calc.cell_size_y/2))
    rect, mask = shape.territory(wall)
    assert mask.get_size() == rect.size
    assert mask.count() > 0
    cx, cy = int(x + calc.cell_size_x/2), int(y + calc.cell_size_y/2)
    assert rect.collidepoint(cx, cy) and mask.get_at((cx - rect.x, cy - rect.y))
    assert shape.territory(wall) is shape.territory(wall)  # cached
    open_cell = next((c, r) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 0)
    assert shape.territory(open_cell) is None


@pytest.mark.perf
def test_open_cell_fast():
    grid = generated(32)
    shape, _, _ = build(grid)
    cells = interior_exposed_walls(grid)[::7][:10]
    times = []
    for cell in cells:
        grid[cell[1]][cell[0]] = 0
        start = time.perf_counter()
        shape.open_cell(cell)
        times.append(time.perf_counter() - start)
    assert statistics.median(times) < 0.010


def test_segment_count_reasonable(maze):
    grid, shape, calc, _ = maze
    square = GridToWallsConverter(calc).convert(grid)
    assert len(shape.segments()) < 6 * len(square)


def _multi_cases(grid, case):
    """Interior wall cells to open together: a run along a wall, two far apart, or two blocks
    added against the rim (the grid is changed for those)."""
    rows, cols = len(grid), len(grid[0])
    walls = interior_exposed_walls(grid)
    if case == 'adjacent':
        return next([(c, r), (c+1, r), (c+2, r)] for c, r in walls
                    if c+2 < cols-2 and grid[r][c+1] == grid[r][c+2] == 1)
    if case == 'apart':
        return [walls[0], walls[-1]]
    r = next(r for r in range(1, rows-3) if grid[r][1] == grid[r+1][1] == 0)
    grid[r][1] = grid[r+1][1] = 1
    return [(1, r), (1, r+1)]


@pytest.mark.parametrize('case', ['adjacent', 'apart', 'rim', 'steel'])
def test_open_cells_matches_fresh_build(case):
    grid = generated()
    rows, cols = len(grid), len(grid[0])
    strong = frozenset()
    if case == 'steel':
        steel, a, b = next(((c, r), (c+1, r), (c+2, r)) for r in range(2, rows-2) for c in range(2, cols-4)
                           if grid[r][c] == grid[r][c+1] == grid[r][c+2] == 1)
        strong, cells = frozenset({steel}), [a, b]
    else:
        cells = _multi_cases(grid, case)
    shape, calc, _ = build(grid, strong)
    before = shape.segments()
    rebuilds = []
    real = shape.field.rebuild
    shape.field.rebuild = lambda: rebuilds.append(1) or real()
    for c, r in cells:
        grid[r][c] = 0
    removed, added = shape.open_cells(cells)
    after = shape.segments()

    fresh, _, _ = build([row[:] for row in grid], strong)
    assert sorted(map(key, after)) == sorted(map(key, fresh.segments()))
    assert len(rebuilds) == 1 and shape.version == 1
    after_ids = {id(s) for s in after}
    assert {id(s) for s in before} - {id(s) for s in removed} | {id(s) for s in added} == after_ids
    assert not any(s.cell in cells for s in after)
    assert cave_checks.watertight_violations(grid, calc, after) == []


def test_open_cell_is_open_cells_of_one():
    grid = generated()
    shape, _, _ = build(grid)
    cell = interior_exposed_walls(grid)[5]
    grid[cell[1]][cell[0]] = 0
    removed, added = shape.open_cell(cell)
    other, _, _ = build([row[:] for row in grid])
    assert sorted(map(key, shape.segments())) == sorted(map(key, other.segments()))
    assert removed and added
