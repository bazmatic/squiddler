import math
from types import SimpleNamespace

import pytest

from physics.sea_snake import MUSCLES, SeaSnakeCommand, SeaSnakePhysics

RADIUS = 10.


def _wave(frame):
    return SeaSnakeCommand(tuple(math.sin(-frame*0.15-i*0.8) for i in range(MUSCLES)))


def _progress(snake, start, heading=0.):
    x, y = snake.centre
    return (x-start[0])*math.cos(heading)+(y-start[1])*math.sin(heading)


@pytest.mark.parametrize('radius', [RADIUS, 30.])
def test_a_travelling_wave_swims_the_snake_forward(radius):
    snake = SeaSnakePhysics((0., 0.), radius=radius)
    start = snake.centre
    for frame in range(120):
        snake.step(_wave(frame), 1.)
    # Swims about 8.7 radii in 120 frames, a wide margin over the 3-radius threshold.
    assert _progress(snake, start) > 3*radius


def test_a_still_snake_does_not_self_propel():
    snake = SeaSnakePhysics((0., 0.), radius=RADIUS)
    start = snake.centre
    for _ in range(120):
        snake.step(SeaSnakeCommand(), 1.)
    assert math.dist(snake.centre, start) < 0.5*RADIUS


def test_command_rejects_wrong_length_and_nan():
    with pytest.raises(ValueError):
        SeaSnakeCommand((0.,)*3)
    with pytest.raises(ValueError):
        SeaSnakeCommand((float('nan'),)*MUSCLES)


def test_walls_stop_the_snake():
    snake = SeaSnakePhysics((0., 0.), radius=RADIUS)
    wall = SimpleNamespace(start=(100., -300.), end=(100., 300.), active=True)
    for frame in range(300):
        snake.step(_wave(frame), 1., walls=[wall])
        assert snake.head[0] < 100.


def test_runaway_speed_raises_runtime_error():
    snake = SeaSnakePhysics((0., 0.), radius=RADIUS)
    snake.impulse((1e12, 0.))
    with pytest.raises(RuntimeError):
        snake.step(SeaSnakeCommand(), 1.)


def _chase(target, frames=400, radius=RADIUS):
    from creatures.sea_snake_control import SeaSnakeController
    snake = SeaSnakePhysics((0., 0.), radius=radius)
    controller = SeaSnakeController()
    start = math.dist(snake.head, target)
    for _ in range(frames):
        snake.step(controller.command(snake, 1., target), 1.)
    return math.dist(snake.head, target) / start


@pytest.mark.parametrize('radius', [RADIUS, 30.])
def test_scripted_controller_closes_distance_to_a_target(radius):
    assert _chase((0., 40*radius), radius=radius) < 0.6


@pytest.mark.parametrize('radius', [RADIUS, 30.])
def test_scripted_controller_turns_toward_a_target_on_the_other_side(radius):
    assert _chase((0., -40*radius), radius=radius) < 0.6


def test_a_bigger_snake_swims_like_a_scaled_up_small_one():
    """Size is pure geometry: three times the radius covers three times the ground and bends the same way."""
    from creatures.sea_snake_control import SeaSnakeController
    small, big = SeaSnakePhysics((0., 0.), radius=10.), SeaSnakePhysics((0., 0.), radius=30.)
    steer, chase = SeaSnakeController(), SeaSnakeController()
    for _ in range(200):
        small.step(steer.command(small, 1., (0., 400.)), 1.)
        big.step(chase.command(big, 1., (0., 1200.)), 1.)
    for (a, b), (c, d) in zip(small.pose(), big.pose()):
        assert math.dist((a[0]*3, a[1]*3), c) < 1. and math.dist((b[0]*3, b[1]*3), d) < 1.


def test_the_catalogue_snake_is_big_and_swims_at_a_chaseable_pace():
    import enemy_kinds
    from creatures.sea_snake_control import SeaSnakeController
    radius = enemy_kinds.kind('sea_snake').tuning['size']
    assert radius >= 25.
    snake = SeaSnakePhysics((0., 0.), radius=radius)
    controller = SeaSnakeController()
    for _ in range(100):
        snake.step(controller.command(snake, 1., (1e6, 0.)), 1.)
    start = snake.centre
    for _ in range(300):
        snake.step(controller.command(snake, 1., (1e6, 0.)), 1.)
    assert 2. <= _progress(snake, start)/300 <= 4.


def test_scripted_controller_idles_without_a_target():
    from creatures.sea_snake_control import SeaSnakeController
    snake = SeaSnakePhysics((0., 0.), radius=RADIUS)
    controller = SeaSnakeController()
    for _ in range(60):
        snake.step(controller.command(snake, 1., None), 1.)
    assert all(math.isfinite(c) for c in snake.centre)
