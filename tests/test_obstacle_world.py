"""The obstacle world: physics gates on pymunk, pushed by movers and stopped by walls and each other."""
import logging
import math
import time
from types import SimpleNamespace

import pytest

import config
from entities.projectile import Projectile
from level_layout import GatePlacement, parse_layout
from maze.config import MazeComplexity
from maze.generator import Maze
from physics.obstacle_world import ObstacleWorld, plank_fits

GRID = 12
PIN = (5, 5)


def make_maze(walls=()):
    layout = parse_layout({'start': [1, 1], 'exit': [10, 10], 'walls': [list(w) for w in walls]}, GRID)
    return Maze(1, complexity=MazeComplexity.EMPTY, grid_size=GRID, layout=layout)


def world_with(placements, walls=()):
    maze = make_maze(walls)
    return maze, ObstacleWorld(maze, placements)


def mover(x, y, vx=0.0, vy=0.0, radius=10.0):
    return SimpleNamespace(x=x, y=y, vx=vx, vy=vy, radius=radius)


def run(world, frames, movers=(), anchored=(), each=None):
    """Move each mover by its own velocity, then step, as the game does once a frame."""
    for _ in range(frames):
        for m in movers:
            m.x += m.vx
            m.y += m.vy
        world.step(1.0, movers, anchored)
        if each is not None:
            each()


def in_wall(maze, point):
    col, row = maze.position_calculator.screen_to_grid(*point)
    return maze.grid[row][col] == 1


def point_segment_distance(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq))
    return math.hypot(p[0] - (ax + dx * t), p[1] - (ay + dy * t))


def segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


def test_pushing_a_free_gate_turns_it():
    maze, world = world_with([GatePlacement(PIN)])
    gate = world.gates[0]
    px, py = gate.pin
    m = mover(px + gate.half_length - 15, py - gate.thickness / 2 - 10 - 6, vy=3.0)
    run(world, 30, [m])
    assert abs(gate.angle) > 0.01
    assert math.hypot(m.vx, m.vy) > 1.5


def test_a_free_gate_runs_down_like_the_ship():
    _, world = world_with([GatePlacement(PIN)])
    gate = world.gates[0]
    gate.angular_velocity = 2.0
    run(world, 60)
    expected = 2.0 * config.SHIP_FRICTION ** 60
    assert abs(gate.angular_velocity - expected) < 0.05 * expected


@pytest.fixture
def real_screen(monkeypatch):
    """The real fullscreen, where cells are wider than tall."""
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)


def maze_of(grid, walls=()):
    layout = parse_layout({'start': [1, 1], 'exit': [grid - 2, grid - 2], 'walls': [list(w) for w in walls]}, grid)
    return Maze(1, complexity=MazeComplexity.EMPTY, grid_size=grid, layout=layout)


# Walls beside a gate at (5, 5) that its plank clears at each angle: along its sides, never across its tips
BESIDE = {0: [(5, 4), (5, 6)], 45: [(6, 4), (4, 6)], 90: [(4, 5), (6, 5)], 135: [(4, 4), (6, 6)]}


@pytest.mark.parametrize('grid', [12, 20])
@pytest.mark.parametrize('angle', [0, 45, 90, 135])
def test_a_gate_beside_walls_stays_put(real_screen, grid, angle):
    maze = maze_of(grid, BESIDE[angle])
    placement = GatePlacement(PIN, angle=angle)
    assert plank_fits(maze, placement)
    world = ObstacleWorld(maze, [placement])
    gate = world.gates[0]
    run(world, 600)
    assert abs(gate.angle - math.radians(angle)) < 1e-6
    assert abs(gate.angular_velocity) < 1e-6


@pytest.mark.parametrize('grid', [12, 20])
@pytest.mark.parametrize('angle', [45, 135])
@pytest.mark.parametrize('wall', [(5, 4), (5, 6)])
def test_a_diagonal_plank_that_would_cut_a_wall_does_not_fit(real_screen, caplog, grid, angle, wall):
    """Cells are wider than tall, so a diagonal plank as long as the average cell runs steeper than the cell's
    diagonal and cuts the cell above or below its tips."""
    maze = maze_of(grid, [wall])
    placement = GatePlacement(PIN, angle=angle, motor='cw' if angle == 135 else None)
    assert not plank_fits(maze, placement)
    with caplog.at_level(logging.WARNING):
        world = ObstacleWorld(maze, [placement, GatePlacement((8, 8))])
    assert [g.placement for g in world.gates] == [GatePlacement((8, 8))]
    assert any('(5, 5)' in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize('grid', [12, 20])
def test_planks_too_close_to_each_other_do_not_fit(real_screen, grid):
    maze = maze_of(grid)
    near = GatePlacement((6, 5), angle=90)      # its plank passes just beyond the tip of the one at (5, 5)
    assert plank_fits(maze, GatePlacement(PIN), [GatePlacement((8, 5), angle=90)])
    assert not plank_fits(maze, GatePlacement(PIN), [near])
    assert not plank_fits(maze, GatePlacement(PIN), [GatePlacement(PIN)])
    world = ObstacleWorld(maze, [GatePlacement(PIN), near])
    assert [g.placement for g in world.gates] == [GatePlacement(PIN)]


def test_a_gate_on_a_cell_kept_clear_does_not_fit(real_screen):
    maze = maze_of(12)
    assert plank_fits(maze, GatePlacement(PIN, angle=45), keep_clear=[(6, 4)])
    assert not plank_fits(maze, GatePlacement(PIN, angle=45), keep_clear=[(5, 6)])


def test_a_gate_stops_against_a_wall():
    maze, world = world_with([GatePlacement(PIN)], walls=[(6, 6)])
    gate = world.gates[0]
    gate.angular_velocity = 3.0

    def check():
        assert not any(in_wall(maze, end) for end in gate.ends())

    run(world, 120, each=check)
    assert gate.angle < math.radians(60)


def test_gates_stop_against_each_other(monkeypatch):
    # On square cells, so the upright plank's tip sits half a cell from the level one: in the spinning one's sweep
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1000)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 1000)
    _, world = world_with([GatePlacement((5, 4)), GatePlacement((6, 6), angle=90)])
    assert len(world.gates) == 2
    top, bottom = world.gates
    top.angular_velocity = 3.0

    def check():
        assert not segments_cross(*top.ends(), *bottom.ends())

    run(world, 120, each=check)
    assert abs(bottom.angle - math.pi / 2) > 0.01  # they met: the spinning gate pushed the other round


def test_a_motor_gate_turns_at_its_rate():
    rate = config.SETTINGS.obstacles.motorRate
    for motor, sign in (('cw', 1), ('ccw', -1)):
        _, world = world_with([GatePlacement(PIN, motor=motor)])
        gate = world.gates[0]
        before = gate.angle
        run(world, 60)
        turned = gate.angle - before
        assert abs(turned - sign * rate) < 0.1 * rate, motor


def test_a_motor_gate_pushes_a_mover_aside():
    rate = config.SETTINGS.obstacles.motorRate
    _, world = world_with([GatePlacement(PIN, motor='cw')])
    gate = world.gates[0]
    px, py = gate.pin
    reach = gate.half_length * 0.6
    start = (px + reach * math.cos(math.radians(30)), py + reach * math.sin(math.radians(30)))
    m = mover(*start)
    run(world, 120, [m])
    assert gate.angle > 0.5 * rate * 2.0  # slowed by the push, but still turning
    assert math.hypot(m.x - start[0], m.y - start[1]) > 20
    assert point_segment_distance((m.x, m.y), *gate.ends()) >= gate.thickness / 2 + m.radius - 2


def squeeze_run(with_mover):
    """A cw motor gate turning down onto wall (5,6), with or without a mover lying on the wall's top face in its
    path. Returns the maze, the gate, the mover and the gate's angle and the mover's gap to the plank each frame."""
    maze, world = world_with([GatePlacement(PIN, motor='cw')], walls=[(5, 6)])
    gate = world.gates[0]
    px, py = gate.pin
    face_y = py + maze.cell_size_y / 2
    # Too big to fit under the plank where it meets the wall's corner, so the plank reaches it first
    m = mover(px + 100, face_y - 16 - 1, radius=16)
    frames = []

    def record():
        gap = point_segment_distance((m.x, m.y), *gate.ends()) - m.radius - gate.thickness / 2
        frames.append((gate.angle, gap))
        values = (m.x, m.y, m.vx, m.vy, gate.angle, gate.angular_velocity)
        assert all(math.isfinite(v) for v in values)
        for wall in maze.walls:
            if wall.active:
                assert point_segment_distance((m.x, m.y), wall.start, wall.end) >= m.radius - 1

    run(world, 120, [m] if with_mover else [], each=record)
    return maze, gate, m, frames


def test_a_squeezed_mover_stays_out_of_walls_and_the_gate_eventually_settles():
    # squeeze_run checks wall clearance and finite physics state throughout the run.
    # Contact solver timing varies; verify the settled outcome rather than exact intermediate angles.
    _, _, _, squeezed = squeeze_run(with_mover=True)
    _, _, _, free = squeeze_run(with_mover=False)
    # Once the mover has slipped out the gate rests against the wall, where it would have stopped anyway
    assert abs(squeezed[-1][0] - squeezed[-31][0]) < 0.01
    assert abs(squeezed[-1][0] - free[-1][0]) < 0.01
    assert squeezed[-1][1] > 0


def test_a_motor_gate_stalls_on_an_anchored_enemy():
    _, world = world_with([GatePlacement(PIN, motor='cw')])
    gate = world.gates[0]
    px, py = gate.pin
    reach = gate.half_length * 0.7
    anemone = mover(px + reach * math.cos(math.radians(40)), py + reach * math.sin(math.radians(40)), radius=14)
    run(world, 120, anchored=[anemone])
    assert (anemone.x, anemone.y) == (px + reach * math.cos(math.radians(40)), py + reach * math.sin(math.radians(40)))
    assert gate.angle < math.radians(40)
    assert point_segment_distance((anemone.x, anemone.y), *gate.ends()) >= gate.thickness / 2 + anemone.radius - 2


def test_a_jammed_motor_gate_frees_when_the_wall_goes():
    maze, world = world_with([GatePlacement(PIN, motor='cw')], walls=[(6, 6)])
    gate = world.gates[0]
    run(world, 60)
    jammed = gate.angle
    assert jammed < math.radians(55)
    run(world, 30)
    assert abs(gate.angle - jammed) < 0.01
    face = next(w for w in maze.walls if w.cell == (6, 6))
    for _ in range(20):
        if maze.damage_wall(face):
            break
    assert (6, 6) not in maze.blocks
    run(world, 90)
    assert gate.angle > math.radians(80)


def test_a_mover_meets_the_plank_where_it_hits_it():
    # Every phase of an 8 px/frame approach: on its contact frame the mover ends at the plank's surface
    for offset in range(8):
        _, world = world_with([GatePlacement(PIN)])
        gate = world.gates[0]
        px, py = gate.pin
        m = mover(px + 5, py - 60 - offset, vy=8.0)
        for _ in range(20):
            m.x += m.vx
            m.y += m.vy
            world.step(1.0, [m])
            if m.vy != 8.0:
                break
        assert m.vy < 0, offset
        gap = point_segment_distance((m.x, m.y), *gate.ends()) - m.radius - gate.thickness / 2
        assert abs(gap) <= 1.5, (offset, gap)


def test_a_fast_ship_does_not_pass_through_a_plank():
    _, world = world_with([GatePlacement(PIN)])
    gate = world.gates[0]
    px, py = gate.pin
    m = mover(px + 5, py - 90, vy=8.0)
    run(world, 40, [m])
    assert m.y < py


def test_a_bullet_nudges_a_gate():
    _, world = world_with([GatePlacement(PIN)])
    gate = world.gates[0]
    px, py = gate.pin
    shot = Projectile((px + gate.half_length - 5, py - gate.thickness / 2 - 1), 90)
    assert world.bullet_hit(shot)
    assert not shot.active
    assert 0 < abs(gate.angular_velocity) < 0.5
    far = Projectile((px, py - gate.half_length * 2), 90)
    assert not world.bullet_hit(far)
    assert far.active


@pytest.mark.parametrize('dt', [1.0, 0.5])
def test_a_fast_shot_does_not_slip_between_frames_through_a_thin_plank(real_screen, dt):
    maze = maze_of(28)
    world = ObstacleWorld(maze, [GatePlacement((14, 14))])
    gate = world.gates[0]
    assert gate.thickness == 6.0
    px, py = gate.pin
    reach = gate.thickness / 2 + 2.0
    speed = 13.0 / dt                                   # 13 px a frame
    # Last frame it was above the plank and now it is below it; neither point touches the plank
    shot = SimpleNamespace(x=px + 10, y=py + reach + 0.5, vx=0.0, vy=speed, radius=2.0, active=True)
    assert shot.y - shot.vy * dt < py - reach
    assert world.bullet_hit(shot, dt)
    assert not shot.active
    assert gate.angular_velocity != 0
    beside = SimpleNamespace(x=px + gate.half_length + 10, y=py + reach + 0.5, vx=0.0, vy=speed, radius=2.0,
                             active=True)
    assert not world.bullet_hit(beside, dt) and beside.active


def test_only_movers_that_touch_a_gate_are_changed():
    _, world = world_with([GatePlacement(PIN)])
    gate = world.gates[0]
    px, py = gate.pin
    far = mover(px + gate.half_length + 120, py, vx=0.5, vy=-0.25)
    # Within reach of the gate, so mirrored into the space, but running alongside the plank without touching it
    near = mover(px - 60, py - gate.thickness / 2 - 10 - 12, vx=1.0)
    for _ in range(60):
        movers = [far, near]
        for m in movers:
            m.x += m.vx
            m.y += m.vy
        before = [(m.x, m.y, m.vx, m.vy) for m in movers]
        world.step(1.0, movers)
        assert [(m.x, m.y, m.vx, m.vy) for m in movers] == before


@pytest.mark.perf
def test_a_crowd_steps_quickly():
    _, world = world_with([GatePlacement(PIN, motor='cw')])
    gate = world.gates[0]
    px, py = gate.pin
    crowd = [mover(px - 200 + i * 36, py + gate.thickness / 2 + 9, vy=-1.0) for i in range(12)]
    began = time.perf_counter()
    run(world, 60, crowd)
    assert time.perf_counter() - began < 0.25


def test_segments_follow_the_plank():
    _, world = world_with([GatePlacement(PIN, motor='cw')])
    gate = world.gates[0]
    frames = round(math.pi / 2 / config.SETTINGS.obstacles.motorRate * config.FPS)
    run(world, frames)
    assert abs(gate.angle - math.pi / 2) < 0.1
    segment = world.segments()[0]
    assert not segment.destructible
    for got, want in zip((segment.start, segment.end), gate.ends()):
        assert math.hypot(got[0] - want[0], got[1] - want[1]) < 1e-6
