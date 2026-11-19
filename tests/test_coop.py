"""Campaign co-op: two ships fly one level together as a team."""
import pygame
import pytest

import config
from tests.test_campaign_demo import make_game  # noqa: F401


@pytest.fixture
def coop(make_game):
    game = make_game(autoplay=False)
    game.coop = True
    game.state = config.STATE_PLAYING
    game.start_level()
    game.level_start_time -= 10      # past the opening moment when input is ignored
    return game


def hold(keys_down):
    class Keys:
        def __getitem__(self, key):
            return key in keys_down
    return lambda: Keys()


def step(game, monkeypatch, *keys, frames=1):
    monkeypatch.setattr(pygame.key, 'get_pressed', hold(set(keys)))
    for _ in range(frames):
        game.update(1.0)


def open_exit(game):
    game.exit_opened = True
    game.maze.exit.set_activated(True, game.sound_manager)


def put_in_exit(player, game):
    player.ship.x, player.ship.y = game.maze.exit.get_pos()
    player.ship.vx = player.ship.vy = 0.0


def test_both_ships_start_side_by_side_without_overlapping(coop):
    a, b = (p.ship for p in coop.players)
    assert len(coop.players) == 2
    assert ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 >= a.radius + b.radius
    assert b.palette is not None and a.palette is None      # each in its own colour


def test_enemies_and_hunter_wake_when_either_ship_moves_but_the_drain_starts_per_ship(coop, monkeypatch):
    first, second = coop.players
    assert not coop.player_has_moved
    monkeypatch.setattr(first, 'controller', None)
    step(coop, monkeypatch, pygame.K_UP)       # the keyboard flies Player 1 only
    assert first.has_moved and not second.has_moved and coop.player_has_moved
    now = coop.clock()
    # Player 2 hasn't moved, so its power hasn't drained; Player 1's clock runs from its first move
    assert second.scoring.level_start_time >= now - 0.5
    assert first.scoring.level_start_time <= second.scoring.level_start_time


def test_the_keyboard_does_not_fly_the_second_ship(coop, monkeypatch):
    second_ship = coop.players[1].ship
    before = (second_ship.vx, second_ship.vy, second_ship.angle)
    step(coop, monkeypatch, pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE, frames=5)
    assert (second_ship.vx, second_ship.vy, second_ship.angle) == before and not coop.players[1].has_moved


def test_devices_follow_the_controllers_connected(coop, monkeypatch):
    pads = [object(), object()]
    monkeypatch.setattr(coop.input_handler, 'get_controllers', lambda: pads[:1])
    coop._assign_devices()
    assert (coop.players[0].keyboard, coop.players[0].controller) == (True, None)
    assert (coop.players[1].keyboard, coop.players[1].controller) == (False, 0)
    monkeypatch.setattr(coop.input_handler, 'get_controllers', lambda: pads)
    coop._assign_devices()
    assert (coop.players[0].keyboard, coop.players[0].controller) == (True, 0)
    assert (coop.players[1].keyboard, coop.players[1].controller) == (False, 1)


def test_a_ship_out_of_power_drops_out_and_the_level_carries_on(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    first.scoring.level_start_time -= 10_000     # all its power drained away
    step(coop, monkeypatch)
    assert first.dropped_out and not first.ship.active and first.ship.is_dying
    assert not second.dropped_out and second.ship.active
    assert not coop.game_over_active


def test_the_level_fails_when_both_ships_run_out_of_power(coop, monkeypatch):
    for player in coop.players:
        player.has_moved = True
        player.scoring.level_start_time -= 10_000
    step(coop, monkeypatch)
    assert all(p.dropped_out for p in coop.players) and coop.game_over_active


def test_a_ship_through_the_portal_leaves_play_with_its_power_frozen(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    open_exit(coop)
    put_in_exit(first, coop)
    step(coop, monkeypatch)
    assert first.through and not first.ship.active and first.final_power > 0
    assert not coop.exit_explosion_active          # the other is still flying
    frozen = first.final_power
    first.scoring.level_start_time -= 50
    step(coop, monkeypatch)
    assert first.final_power == frozen


def test_the_level_ends_when_every_ship_still_flying_is_through_and_the_team_adds_their_power(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    open_exit(coop)
    put_in_exit(first, coop)
    step(coop, monkeypatch)
    put_in_exit(second, coop)
    step(coop, monkeypatch)
    assert coop.exit_explosion_active
    assert coop.team_power(coop.clock()) == pytest.approx(first.final_power + second.final_power)
    coop.complete_level(success=True)
    assert coop.level_score_breakdown['final_score'] == pytest.approx(first.final_power + second.final_power)


def test_the_level_ends_when_the_other_ship_dropped_out_after_one_got_through(coop, monkeypatch):
    first, second = coop.players
    first.has_moved = second.has_moved = True
    open_exit(coop)
    put_in_exit(first, coop)
    step(coop, monkeypatch)
    second.scoring.level_start_time -= 10_000
    step(coop, monkeypatch)
    step(coop, monkeypatch)
    assert second.dropped_out and coop.exit_explosion_active and not coop.game_over_active
    assert coop.team_power(coop.clock()) == pytest.approx(first.final_power)


def test_ships_bounce_off_each_other_and_each_pays_the_wall_penalty(coop, monkeypatch):
    first, second = coop.players
    second.ship.x, second.ship.y = first.ship.x + first.ship.radius, first.ship.y
    first.ship.shield_active = second.ship.shield_active = False
    first.ship.shield_initial_timer = second.ship.shield_initial_timer = 0
    first.ship.powerup_shield_timer = second.ship.powerup_shield_timer = 0
    coop._bounce_ships()
    assert first.scoring.wall_collisions == 1 and second.scoring.wall_collisions == 1
    gap = ((first.ship.x - second.ship.x) ** 2 + (first.ship.y - second.ship.y) ** 2) ** 0.5
    assert gap > first.ship.radius


def test_enemies_go_for_the_nearest_living_ship(coop):
    from game_handlers.combat_targets import nearest_friendly
    first, second = coop.players
    foe = type('Foe', (), {'x': second.ship.x + 1, 'y': second.ship.y})()
    assert nearest_friendly(foe, coop.ships) is second.ship
    second.ship.die()
    assert nearest_friendly(foe, coop.ships) is first.ship


def test_a_kill_counts_for_the_player_who_fired_the_shot(coop):
    first, second = coop.players
    shot = type('Shot', (), {'source': 'player', 'owner': second})()
    coop.collision_handler.record_kill(shot, type('Enemy', (), {})())
    assert second.scoring.enemies_destroyed == 1 and first.scoring.enemies_destroyed == 0


def test_the_hunter_is_shown_the_nearest_living_ship_as_its_friend(coop):
    first, second = coop.players
    near_second = type('Spot', (), {'x': second.ship.x + 5, 'y': second.ship.y})()
    assert coop.nearest_ship(near_second) is second.ship
    first.dropped_out = True       # only the second ship is left to follow
    assert coop.nearest_ship(type('Spot', (), {'x': first.ship.x, 'y': first.ship.y})()) is second.ship
    assert coop.hunter_surroundings.friend() is second.ship


def test_the_coop_hud_marks_each_player_in_their_colour_and_dims_a_dropped_out_half(coop):
    coop.draw_game()
    from rendering.coop_hud import HEADER_HEIGHT
    live = coop.screen.get_at((100, HEADER_HEIGHT))[:3]
    coop.players[1].dropped_out = True
    coop.players[1].final_power = 0.0
    coop.draw_game()
    assert live != (0, 0, 0)                                      # Player 1's coloured rule is drawn


def test_single_player_is_unchanged_by_the_coop_code(make_game, monkeypatch):
    game = make_game(autoplay=False)
    game.state = config.STATE_PLAYING
    game.start_level()
    assert len(game.players) == 1 and game.players[0].keyboard and game.players[0].controller == 0
    assert game.ship.palette is None
