"""The Maze plays on curved cave walls: CaveShape's segments, damaged per block, rebuilt on a destroy."""
import copy
import math
import pathlib
import random
import statistics
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pygame
import pytest

import config
import level_config
from entities.explosion import Blast
from game_handlers.collision_handler import CollisionHandler
from maze.config import MazeComplexity
from maze.converter import SIDES
from maze.generator import Maze
from maze.wall_segment import WallSegment
from physics.thrust import sweep_walls
from scoring.system import ScoringSystem
from tests import cave_checks
from tests.test_hunter_integration import game  # noqa: F401 - pytest fixture
from tests.test_layout_play import REAL, real_level_config

LEVELS = sorted(int(p.stem) for p in pathlib.Path(__file__).resolve().parent.parent.glob('levels/*.json'))
SEEDS_PER_PRESET = 10  # per screen size: the sweep runs on two


def key(segment):
    return (tuple(round(v, 3) for v in segment.start), tuple(round(v, 3) for v in segment.end), segment.cell)


def keys(walls):
    return sorted(key(w) for w in walls)


def maze_with(*cells, grid=10):
    maze = Maze(1, MazeComplexity.EMPTY, grid)
    for x, y in cells:
        maze.grid[y][x] = 1
    maze.rebuild_walls()
    return maze


def faces(maze, cell):
    return [w for w in maze.walls if w.cell == cell]


def level_maze(level):
    """A level's maze as the game builds it: the real level config, the global seed set first."""
    seed = REAL['get_level_seed'](level)
    random.seed(seed)
    return Maze(level, complexity=REAL['get_maze_complexity'](level), grid_size=REAL['get_maze_grid_size'](level),
                layout=level_config.get_level_layout(level), seed=seed)


def generated_maze(complexity, size, seed, level=1):
    random.seed(seed)
    return Maze(level, complexity=complexity, grid_size=size, seed=seed)


def indexed(maze):
    """Every wall object the spatial grid can return."""
    grid = maze.spatial_grid
    return [grid.walls[i] for i in set().union(*(cell for row in grid.grid for cell in row))]


def assert_index_consistent(maze):
    found = indexed(maze)
    assert all(w.active for w in found)
    assert {id(w) for w in found} == {id(w) for w in maze.walls}


def test_maze_is_curved_by_default():
    maze = Maze(1, MazeComplexity.EMPTY, 10)
    assert maze.curved and maze.cave is not None
    assert any(abs(w.start[0] - w.end[0]) > 1e-6 and abs(w.start[1] - w.end[1]) > 1e-6 for w in maze.walls)


def test_square_mode_unchanged():
    maze = Maze(1, MazeComplexity.EMPTY, 10, curved=False)
    assert maze.cave is None
    assert keys(maze.walls) == keys(maze.converter.convert(maze.grid))


def test_block_shares_hit_points_across_segments():
    maze = maze_with((4, 4), (5, 4))
    segments = faces(maze, (4, 4))
    assert len(segments) > 1
    assert maze.damage_wall(segments[0]) is False
    assert all(s.hit_points == config.WALL_HIT_POINTS - 1 for s in faces(maze, (4, 4)))
    assert maze.blocks[(4, 4)] == config.WALL_HIT_POINTS - 1


def test_destroy_takes_wall_hit_points_hits():
    maze = maze_with((4, 4), (5, 4))
    generation = maze.wall_generation
    before = {id(w) for w in maze.walls}
    for i in range(config.WALL_HIT_POINTS):
        assert (4, 4) in maze.blocks
        destroyed = maze.damage_wall(faces(maze, (4, 4))[0])
        assert destroyed == (i == config.WALL_HIT_POINTS - 1)
    assert (4, 4) not in maze.blocks
    assert maze.grid[4][4] == 0
    assert maze.wall_generation == generation + 1
    assert faces(maze, (4, 4)) == []
    new = [w for w in maze.walls if id(w) not in before]
    assert new  # (5, 4)'s side that looked onto (4, 4)
    for w in new:
        assert w.cell == (5, 4) and w.hit_points == maze.blocks[(5, 4)]
        assert any(f is w for f in maze.spatial_grid.get_walls_along_path(w.start, w.end, 1.0))
    assert_index_consistent(maze)


def test_rim_never_destroyed():
    maze = Maze(1, MazeComplexity.EMPTY, 10)
    rim = set(maze.blocks)
    before = keys(maze.walls)
    for wall in list(maze.walls):
        for _ in range(config.WALL_HIT_POINTS * 3):
            maze.damage_wall(wall)
    assert set(maze.blocks) == rim
    assert keys(maze.walls) == before
    assert all(w.active for w in maze.walls)
    assert all(hp >= 1 for hp in maze.blocks.values())


def _destroy_chain(maze, count):
    """Interior destructible blocks, each next to the one before, destroyed one at a time."""
    grid = maze.grid
    rows, cols = len(grid), len(grid[0])
    interior = lambda c, r: 1 <= c < cols-1 and 1 <= r < rows-1 and grid[r][c] == 1 \
        and not maze.converter.on_perimeter(grid, c, r) and (c, r) not in maze.strong_cells
    start = next(w.cell for w in maze.walls if interior(*w.cell))
    chain, frontier = [], [start]
    while frontier and len(chain) < count:
        cell = frontier.pop(0)
        if cell in chain or not interior(*cell):
            continue
        chain.append(cell)
        frontier.extend((cell[0]+dx, cell[1]+dy) for dx, dy in SIDES)
    assert len(chain) == count
    return chain


def test_repeated_opens_match_fresh_build_and_keep_spatial_index():
    maze = generated_maze(MazeComplexity.COMPLEX, 18, seed=4242)
    for cell in _destroy_chain(maze, 5):
        wall = faces(maze, cell)[0]
        assert maze.damage_wall(wall, damage=config.WALL_HIT_POINTS) is True
        fresh = generated_maze(MazeComplexity.COMPLEX, 18, seed=4242)
        fresh.grid = copy.deepcopy(maze.grid)
        fresh.rebuild_walls()
        assert keys(maze.walls) == keys(fresh.walls)
        assert_index_consistent(maze)
        assert all(w.hit_points == maze.blocks[w.cell] for w in maze.walls)


def test_owner_only_change_keeps_spatial_index():
    """A rebuild can hand a segment to another block without moving it: the removed and the added
    segment then have the same ends, which is all WallSegment equality (and the spatial grid) sees."""
    maze = generated_maze(MazeComplexity.COMPLEX, 18, seed=4242)
    first = _destroy_chain(maze, 1)[0]
    real_open = maze.cave.open_cell
    swapped = []

    def open_with_owner_change(cell):
        removed, added = real_open(cell)
        gone = {id(w) for w in removed}
        survivor = next(w for w in maze.walls if w.cell != cell and id(w) not in gone)
        other = next(c for c in maze.blocks if c != survivor.cell)
        copy_ = WallSegment(survivor.start, survivor.end, maze.blocks[other], survivor.destructible, other)
        swapped.append(copy_)
        return removed + [survivor], added + [copy_]
    maze.cave.open_cell = open_with_owner_change

    assert maze.damage_wall(faces(maze, first)[0], damage=config.WALL_HIT_POINTS) is True
    assert swapped and swapped[0] in maze.walls
    assert any(w is swapped[0] for w in maze.spatial_grid.get_nearby_walls(swapped[0].start, 1.0))
    assert_index_consistent(maze)


def _clearance(maze, pos):
    """px from pos to the nearest wall segment, and to the nearest wall cell's square (the outline
    square walls would have there)."""
    import numpy as np
    ends = np.array([(w.start[0], w.start[1], w.end[0], w.end[1]) for w in maze.walls], float)
    a, d = ends[:, :2], ends[:, 2:] - ends[:, :2]
    p = np.array(pos, float)
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    outline = float(np.hypot(*(a + d * t[:, None] - p).T).min())
    calc = maze.position_calculator
    square = math.inf
    for x, y in maze.blocks:
        left, top = calc.grid_to_screen(x, y)
        gap_x = max(0.0, left - pos[0], pos[0] - (left + calc.cell_size_x))
        gap_y = max(0.0, top - pos[1], pos[1] - (top + calc.cell_size_y))
        square = min(square, math.hypot(gap_x, gap_y))
    return outline, square


def _layout_spawns(maze):
    """(what, pixel position, radius) for everything a layout places: the ship, the exit, the beacon
    and each creature, at its cell's centre."""
    from entities.jev_beacon import JevBeacon
    import enemy_kinds
    layout, calc = maze.layout, maze.position_calculator
    out = [('ship', maze.start_pos, config.SHIP_SIZE), ('exit', maze.exit.get_pos(), maze.exit.radius)]
    if layout.beacon is not None:
        out.append(('beacon', calc.grid_center_to_screen(*layout.beacon),
                    JevBeacon.RING_RADIUS * config.POWERUP_CRYSTAL_SIZE))
    out.extend((f'{kind} {cell}', calc.grid_center_to_screen(*cell), enemy_kinds.kind(kind).tuning['size'])
               for kind, cell in layout.enemies)
    return out


# The outline may sit this far inside a disc whose clearance the cell's square walls only just give
SPAWN_TOLERANCE = 0.5


def test_layout_spawns_keep_their_disc_clear(monkeypatch):
    """On the real display, curved rock leaves every hand-placed spawn's whole disc clear: as clear
    as the square walls would (which already cut into a few big discs: the exit portal, a large
    creature in a narrow cell)."""
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)
    short = []
    for level in LEVELS:
        maze = level_maze(level)
        if maze.layout is None:
            continue
        assert maze.cave is not None
        for what, pos, radius in _layout_spawns(maze):
            outline, square = _clearance(maze, pos)
            if maze.cave.is_rock(pos) or outline < min(radius, square) - SPAWN_TOLERANCE:
                short.append((level, what, radius, round(outline, 1), round(square, 1)))
    assert short == []


@pytest.mark.parametrize('screen', [(3840, 2160), (1352, 878)])
def test_generated_spawns_keep_their_disc_clear(monkeypatch, screen):
    """A generated maze's ship start, exit and enemy spawns (which is_open checks with a 15 px disc)."""
    monkeypatch.setattr(config, 'SCREEN_WIDTH', screen[0])
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', screen[1])
    short = []
    for seed in range(4):
        for complexity in (MazeComplexity.NORMAL, MazeComplexity.EXTREME):
            maze = generated_maze(complexity, 32, seed)
            random.seed(3)
            spawns = [('ship', maze.start_pos, config.SHIP_SIZE), ('exit', maze.exit.get_pos(), maze.exit.radius)]
            spawns += [('enemy', pos, 15) for pos in maze.get_valid_spawn_positions(30)]
            for what, pos, radius in spawns:
                outline, square = _clearance(maze, pos)
                if maze.cave.is_rock(pos) or outline < min(radius, square) - SPAWN_TOLERANCE:
                    short.append((seed, complexity.name, what, round(outline, 1), round(square, 1)))
    assert short == []


def _outline_shape(maze):
    """(share of the outline's length that is neither horizontal nor vertical, radians the outline
    turns per cell of its length). Wiggles turn a lot for little length; long sweeping curves
    turn little."""
    unit = math.sqrt(maze.cell_size_x * maze.cell_size_y)
    point = lambda p: (round(p[0], 3), round(p[1], 3))
    starts = {point(w.start): w for w in maze.walls}
    total = slanted = turned = 0.0
    for w in maze.walls:
        dx, dy = w.end[0] - w.start[0], w.end[1] - w.start[1]
        length = math.hypot(dx, dy)
        total += length
        if min(abs(dx), abs(dy)) > 0.05 * length:
            slanted += length
        after = starts.get(point(w.end))
        if after is not None and length:
            ex, ey = after.end[0] - after.start[0], after.end[1] - after.start[1]
            turned += abs(math.atan2(dx*ey - dy*ex, dx*ex + dy*ey))
    return slanted / total, turned / total * unit


# level: (least slanted share, most turning per cell), on the real display. Measured 2026-10-26
# with the tuning in settings.json (L13 0.73/0.83, L21 0.74/0.65, L26 0.81/0.90, L27 0.34/0.37);
# floors about 87% of those, ceilings 15% over. The rough two-octave tuning before turned
# 1.4 rad per cell on L13 and L26.
OUTLINE_SHAPE = {13: (0.64, 0.95), 21: (0.64, 0.75), 26: (0.70, 1.05), 27: (0.30, 0.43)}


def test_curved_rock_is_smooth_large_scale_curve(monkeypatch):
    """Curved rock sweeps in long gentle curves: plenty of it is slanted (it is curved at all), but
    it turns little per length (no small wiggles)."""
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)
    found = {level: tuple(round(v, 2) for v in _outline_shape(level_maze(level))) for level in OUTLINE_SHAPE}
    assert all(found[level][0] >= least and found[level][1] <= most
               for level, (least, most) in OUTLINE_SHAPE.items()), found


def test_is_open_rejects_rock():
    maze = maze_with((4, 4))
    centre = maze.position_calculator.grid_center_to_screen(4, 4)
    assert maze.cave.is_rock(centre)
    # A tiny circle deep inside the block's rock touches no segment, but is still not open
    maze.blocks.pop((4, 4))  # only the rock test is left to reject it
    assert maze.is_open(centre, 1) is False


def test_ship_bounces_off_the_curve():
    maze = generated_maze(MazeComplexity.NORMAL, 18, seed=77)
    radius = config.SHIP_SIZE
    for wall in maze.walls:
        dx, dy = wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]
        length = math.hypot(dx, dy)
        if length < 4 or min(abs(dx), abs(dy)) < 0.3 * length:
            continue
        n = (dy / length, -dx / length)  # rock is on the right, so this points into open space
        t = (dx / length, dy / length)
        mid = ((wall.start[0] + wall.end[0]) / 2, (wall.start[1] + wall.end[1]) / 2)
        contact = (mid[0] + n[0]*radius, mid[1] + n[1]*radius)
        v = (-n[0]*4 + t[0]*1.5, -n[1]*4 + t[1]*1.5)
        path = [(contact[0] + v[0]*k, contact[1] + v[1]*k) for k in (-1, -0.5, 0, 0.5, 1)]
        from utils.math_utils import circle_line_collision
        if any(circle_line_collision(p, radius + 1, o.start, o.end) for o in maze.walls if o is not wall
               for p in path):
            continue
        body = SimpleNamespace(prev_x=path[0][0], prev_y=path[0][1], x=path[-1][0], y=path[-1][1],
                               vx=v[0], vy=v[1], radius=radius)
        assert sweep_walls(body, maze.walls, maze.spatial_grid)
        dot = v[0]*n[0] + v[1]*n[1]
        expected = ((v[0] - 2*dot*n[0]) * config.COLLISION_RESTITUTION,
                    (v[1] - 2*dot*n[1]) * config.COLLISION_RESTITUTION)
        assert abs(body.vx - expected[0]) < 1e-6 and abs(body.vy - expected[1]) < 1e-6
        return
    pytest.fail('no clear, slanted segment to bounce off')


def _anti_goal_violations(maze):
    field = maze.cave.field
    return {
        'links': cave_checks.centre_and_link_violations(field),
        'islands': cave_checks.rock_island_violations(field),
        'gates': cave_checks.gate_swing_violations(field),
        'cores': cave_checks.wall_core_violations(field),
        'steel': cave_checks.steel_face_violations(field),
        'watertight': cave_checks.watertight_violations(maze.grid, maze.position_calculator, maze.walls),
    }


SCREENS = {'default 3840x2160': (3840, 2160), 'real display 1352x878': (1352, 878)}


# One case per level and per generator preset and size, so a parallel run spreads the sweep over its workers.
SWEEP = [f'level {level}' for level in LEVELS] + [f'{c.name} {size}' for c in MazeComplexity for size in (18, 32)]


def _sweep_mazes(case):
    if case.startswith('level '):
        level = int(case.split()[1])
        return [(case, lambda: level_maze(level))]
    name, size = case.split()
    complexity, size = MazeComplexity[name], int(size)
    return [(f'{case} seed {seed}', lambda sd=seed: generated_maze(complexity, size, sd, level=sd + 1))
            for seed in range(SEEDS_PER_PRESET)]


@pytest.mark.parametrize('case', SWEEP)
@pytest.mark.parametrize('screen', SCREENS)
def test_all_levels_meet_anti_goals(monkeypatch, screen, case):
    """Every level and SEEDS_PER_PRESET generated mazes per preset and size, on the default screen
    and on the real fullscreen display (where cells are about a third the size)."""
    width, height = SCREENS[screen]
    monkeypatch.setattr(config, 'SCREEN_WIDTH', width)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', height)
    failures = {}
    for name, build in _sweep_mazes(case):
        bad = {check: found[:3] for check, found in _anti_goal_violations(build()).items() if found}
        if bad:
            failures[name] = bad
    assert failures == {}


def _cluster_maze():
    random.seed(5)  # the same start and exit corners every time: the rock keeps them clear
    maze = Maze(1, MazeComplexity.EMPTY, 32)
    cluster = [(10, 10), (11, 10), (10, 11), (11, 11)]
    for x, y in cluster:
        maze.grid[y][x] = 1
    maze.rebuild_walls()
    return maze, cluster


@pytest.mark.perf
def test_multi_block_blast_budget():
    scoring = ScoringSystem()
    scoring.start_level(1)
    handler = CollisionHandler(Mock(), scoring, Mock())
    times = []
    for _ in range(3):
        maze, cluster = _cluster_maze()
        left, top = maze.position_calculator.grid_to_screen(10, 10)
        centre = (left + maze.cell_size_x, top + maze.cell_size_y)
        blast = Blast(centre, maze.cell_size * 1.6, wall_hits=config.WALL_HIT_POINTS)
        generation = maze.wall_generation
        began = time.perf_counter()
        handler.handle_blast(blast, None, None, [], lambda shot: False, maze)
        times.append(time.perf_counter() - began)
        assert all(cell not in maze.blocks for cell in cluster)
        assert maze.wall_generation == generation + 4
        assert_index_consistent(maze)
    assert min(times) < 0.040, times


def test_blast_damages_each_block_wall_hits_times():
    """A block with many segments in reach still takes the blast's wall hits once, not once per segment."""
    scoring = ScoringSystem()
    scoring.start_level(1)
    handler = CollisionHandler(Mock(), scoring, Mock())
    maze, cluster = _cluster_maze()
    left, top = maze.position_calculator.grid_to_screen(10, 10)
    centre = (left + maze.cell_size_x, top + maze.cell_size_y)
    handler.handle_blast(Blast(centre, maze.cell_size * 1.6, wall_hits=1), None, None, [], lambda shot: False, maze)
    assert all(maze.blocks[cell] == config.WALL_HIT_POINTS - 1 for cell in cluster)


def test_rail_shot_destroys_a_curved_block_whole():
    from weapons.projectile import RailProjectile
    maze = maze_with((4, 4), (5, 4))
    wall = faces(maze, (4, 4))[0]
    shot = RailProjectile.__new__(RailProjectile)
    shot.x, shot.y = wall.start
    shot.active = True
    shot.hit_surface((1.0, 0.0), wall, maze, None)
    assert (4, 4) not in maze.blocks
    assert shot.active  # it keeps going through the destroyed block
    assert_index_consistent(maze)


@pytest.mark.parametrize('level', [4, 20, 22])
def test_editor_preview_matches_play(game, monkeypatch, level):  # noqa: F811
    from rendering.level_editor import LevelEditor
    real_level_config(monkeypatch)
    screen = pygame.display.set_mode((1352, 878))
    editor = LevelEditor(game, screen)
    editor._open_level(level)
    preview, _ = editor._preview_scene()
    assert keys(preview.walls) == keys(level_maze(level).walls)


def _episode_seconds(curved, level, horizon):
    from experiments import harness
    began = time.perf_counter()
    harness.run_episode(level, 0, horizon=horizon, curved_walls=curved)
    seconds = time.perf_counter() - began
    assert (harness._game.maze.cave is not None) == curved, (level, curved)  # really the walls asked for
    return seconds


# (level, episode seconds, the most curved rock may cost against square walls). Level 27, the
# farming level, is crowded. Level 13 has the hunter, whose sight of curved rock costs most: it
# measured 1.36-1.39x (6 s episodes, best of two, 2026-10-26), so its cap is 1.25 times that.
FRAME_TIME_GATES = [(27, 3.0, 1.3), (13, 6.0, 1.75)]


@pytest.mark.perf
@pytest.mark.parametrize('level, horizon, limit', FRAME_TIME_GATES)
def test_curved_walls_cost_little_frame_time(level, horizon, limit):
    """A level plays about as fast on curved rock as on square walls.

    Relative, at the real screen size the harness plays at. One untimed short run of each first (the
    harness builds its Game once per process), then the two modes alternate, best of two each."""
    from experiments import harness
    try:
        for curved in (False, True):
            _episode_seconds(curved, level, 0.5)
        times = {False: math.inf, True: math.inf}
        for curved in (False, True, True, False):
            times[curved] = min(times[curved], _episode_seconds(curved, level, horizon))
    finally:
        harness.reset()  # the harness's Game changes the screen size for the whole process
    assert times[True] <= limit * times[False], times


def _destroy_times(curved, cells, count=12):
    """Seconds per destroy of the same blocks through Maze.damage_wall, and per CaveShape.open_cell."""
    random.seed(9)
    maze = Maze(1, complexity=MazeComplexity.COMPLEX, grid_size=32, seed=9, curved=curved)
    opens = []
    if maze.cave is not None:
        real_open = maze.cave.open_cell

        def timed_open(cell):
            began = time.perf_counter()
            result = real_open(cell)
            opens.append(time.perf_counter() - began)
            return result
        maze.cave.open_cell = timed_open
    destroys = []
    for cell in cells:
        if len(destroys) == count:
            break
        exposed = faces(maze, cell)
        if not exposed:
            continue
        began = time.perf_counter()
        assert maze.damage_wall(exposed[0], damage=config.WALL_HIT_POINTS)
        destroys.append(time.perf_counter() - began)
    assert len(destroys) == count
    return statistics.median(destroys), (statistics.median(opens) if opens else 0.0)


@pytest.mark.perf
def test_destroying_blocks_stays_within_budget(monkeypatch):
    """Grenades and rails destroy blocks: on the real screen, the same 12 blocks of a grid-32 maze,
    curved against square. A curved destroy rebuilds the rock (CaveShape.open_cell, whose own budget
    is 10 ms in test_cave_shape); the Maze's bookkeeping around it stays close to a square destroy,
    and the whole destroy within the brief's 10 ms a block (40 ms for a 4-block blast)."""
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)
    random.seed(9)
    square_maze = Maze(1, complexity=MazeComplexity.COMPLEX, grid_size=32, seed=9, curved=False)
    cells = [c for c in sorted(square_maze.blocks) if not square_maze.converter.on_perimeter(square_maze.grid, *c)]
    random.Random(1).shuffle(cells)
    _destroy_times(True, cells, count=2)  # warm up
    square, _ = _destroy_times(False, cells)
    curved, open_cell = _destroy_times(True, cells)
    assert curved - open_cell <= 2.0 * square + 0.0005, (curved, open_cell, square)
    assert curved < 0.010, (curved, open_cell, square)


def test_blast_rebuilds_the_rock_once(monkeypatch):
    """A grenade taking out 4 adjacent blocks on the real screen: one field rebuild for the whole
    blast, within a frame-friendly budget, with the same hits and a consistent index."""
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)
    scoring = ScoringSystem()
    scoring.start_level(1)
    handler = CollisionHandler(Mock(), scoring, Mock())
    times = []
    for _ in range(5):
        maze, cluster = _cluster_maze()
        rebuilds = []
        real = maze.cave.field.rebuild
        maze.cave.field.rebuild = lambda: rebuilds.append(1) or real()
        left, top = maze.position_calculator.grid_to_screen(10, 10)
        centre = (left + maze.cell_size_x, top + maze.cell_size_y)
        generation = maze.wall_generation
        began = time.perf_counter()
        handler.handle_blast(Blast(centre, maze.cell_size * 1.6, wall_hits=config.WALL_HIT_POINTS),
                             None, None, [], lambda shot: False, maze)
        times.append(time.perf_counter() - began)
        assert len(rebuilds) == 1
        assert all(cell not in maze.blocks for cell in cluster)
        assert maze.wall_generation > generation
        assert_index_consistent(maze)
        fresh, _ = _cluster_maze()
        fresh.grid = copy.deepcopy(maze.grid)
        fresh.rebuild_walls()
        assert keys(maze.walls) == keys(fresh.walls)
    assert statistics.median(times) < 0.012, times


def test_batched_destroy_skips_stale_segments():
    """Inside a batch a destroyed block's old segments stay until the end; hitting them again does nothing."""
    maze, cluster = _cluster_maze()
    stale = faces(maze, cluster[0])
    with maze.batched_destroys():
        assert maze.damage_wall(stale[0], damage=config.WALL_HIT_POINTS) is True
        assert maze.damage_wall(stale[-1], damage=config.WALL_HIT_POINTS) is False
        assert maze.blocks[cluster[1]] == config.WALL_HIT_POINTS
    assert faces(maze, cluster[0]) == []
    assert all(not w.active for w in stale)
    assert_index_consistent(maze)


def _harness_maze_cave(**kw):
    from experiments import harness
    seen = []
    harness.run_episode(13, 0, horizon=0.1, trace=lambda t, g, h: seen.append(g.maze.cave), **kw)
    harness.reset()
    return seen[0]


def test_harness_defaults_to_square_walls():
    assert _harness_maze_cave() is None


def test_harness_curves_walls_when_asked():
    assert _harness_maze_cave(curved_walls=True) is not None


def test_new_run_records_curved_walls_and_old_runs_stay_square(tmp_path):
    import json
    from tools import ppo_neural as ppo
    args = ppo.build_parser().parse_args(['--out', str(tmp_path), '--levels', '25'])
    assert ppo.shaping_from(args)['curved_walls'] is True
    ppo.write_run_config(args)
    assert json.loads((tmp_path / 'config.json').read_text())['rules']['curved_walls'] is True


def test_resuming_a_run_without_the_record_stays_square(tmp_path, monkeypatch):
    import json
    pytest.importorskip('torch')
    from hunter.neural.policy import random_policy
    from tests.test_ppo_loop import fake_rollout_factory
    from tools import ppo_neural as ppo
    start = tmp_path / 'start.npz'
    random_policy(0, hidden=16, embed=8).save(start)
    out = tmp_path / 'old'
    ppo.train_rl(start, out, iterations=1, episodes=6, workers=1, rollout=fake_rollout_factory(), log=lambda *_: None)
    (out / 'config.json').write_text(json.dumps({'episode': {'levels': [25]}, 'shaping': {}}))
    seen = []

    class Spy:
        def __init__(self, workers, shaping=None, job=None):
            seen.append(shaping)
            self.inner = fake_rollout_factory()

        def __call__(self, jobs):
            return self.inner(jobs, 1)

        def close(self):
            pass

    monkeypatch.setattr(ppo, 'PoolRollout', Spy)
    ppo.train_rl(start, out, iterations=2, episodes=6, workers=1, levels=[25], shaping={'curved_walls': True},
                 log=lambda *_: None, resume=True)
    assert seen[0]['curved_walls'] is False
