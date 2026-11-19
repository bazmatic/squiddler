"""The sea snake playtest: free-swimming snakes in open water."""
import math

import config
from tests.test_level_arc import game  # shared isolated Game fixture


def test_sea_snake_playtest_runs_in_real_game_loop(game, monkeypatch):
    from game_handlers.spawn_manager import SpawnManager
    seated = []
    original = SpawnManager._place_layout_crawler
    monkeypatch.setattr(SpawnManager, '_place_layout_crawler',
                        lambda self, crawler, maze: (seated.append(crawler), original(self, crawler, maze))[1])
    game.level = 35
    game.start_level()
    game.state = config.STATE_PLAYING
    game.player_has_moved = True
    game.ship.game_started = True
    snakes = list(game.roster.of_kind('sea_snake'))
    assert len(snakes) == 3
    assert not [s for s in seated if s in snakes]        # swimmers are not planted on walls
    for _ in range(600):
        game.update(1.)
    assert all(math.isfinite(c.x) and math.isfinite(c.y) for c in snakes)


def test_the_exit_stays_locked_until_every_snake_is_dead(game):
    game.level = 35
    game.start_level()
    game.state = config.STATE_PLAYING
    game.player_has_moved = True
    game.ship.game_started = True
    snakes = [*game.roster.of_kind('sea_snake'), *game.roster.of_kind('mini_sea_snake'),
              *game.roster.of_kind('teeny_sea_snake')]
    assert len(snakes) == 12
    game.update(1.)
    assert not game.exit_opened
    for snake in snakes:
        snake.die()
    for _ in range(300):
        game.update(1.)
    assert game.exit_opened


def test_mini_sea_snakes_are_small_fragile_and_swim_on_the_same_brain():
    import creatures
    from tests.creature_parts import encounter, open_maze, world
    made = {}
    for kind in ('sea_snake', 'mini_sea_snake', 'teeny_sea_snake'):
        c = creatures.build(kind, (400., 286.), angle=0.)
        encounter(c, world_=world(maze=open_maze([]))).ready()
        made[kind] = c
    big, mini = made['sea_snake'], made['mini_sea_snake']
    assert mini.radius < big.radius*.7
    assert mini.hit_points < big.hit_points
    teeny = made['teeny_sea_snake']
    assert teeny.radius < mini.radius and teeny.hit_points == 1


def test_level_35_has_minis_among_the_snakes(game):
    game.level = 35
    game.start_level()
    assert len(list(game.roster.of_kind('mini_sea_snake'))) == 4
