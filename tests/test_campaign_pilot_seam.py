"""A ship pilot flies the player ship in Demo, on a clock the game owns."""
import math

import pygame
import pytest
from unittest.mock import Mock

import config
import level_config
from hunter.campaign_pilot import ShipCommand


class Pilot:
    def __init__(self, command):
        self.command = command

    def decide(self, game):
        return self.command


@pytest.fixture
def game(tmp_path, monkeypatch):
    from game import Game
    from profiles import ProfileManager

    pygame.init()
    screen = pygame.display.set_mode((1352, 878))
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)
    monkeypatch.setattr(config, 'SPLASH_ENABLED', False)
    monkeypatch.delenv('START_LEVEL', raising=False)
    manager = ProfileManager(tmp_path / 'profiles.json')
    monkeypatch.setitem(Game.__init__.__globals__, 'ProfileManager', lambda: manager)
    monkeypatch.setitem(Game.__init__.__globals__, 'SoundManager', Mock)
    monkeypatch.setattr(level_config, 'level_has_hunter', lambda _: False)
    game = Game(screen)
    game.autoplay = object()  # Demo: nobody at the keyboard
    game.levels_ended = []   # what the game hands Demo when the ship leaves a level
    monkeypatch.setattr(game, '_autoplay_level_over',
                        lambda reason=None: reason is not None and (game.levels_ended.append(reason) or True))
    game.clock = lambda: game.fake_now
    game.clock_ms = lambda: int(game.fake_now * 1000)
    game.fake_now = 1000.0
    game.state = config.STATE_PLAYING
    game.start_level()
    game.fake_now += 1.0  # past the post-start input grace
    yield game
    game._close_hunter()


def test_a_ship_pilot_flies_the_ship_and_the_keyboard_does_not(game, monkeypatch):
    monkeypatch.setattr(pygame.key, 'get_pressed',
                        lambda: type('K', (), {'__getitem__': lambda s, k: k == pygame.K_LEFT})())
    game.ship_pilot = Pilot(ShipCommand(turn=1, thrust=True, fire=False))
    game.keys_pressed_at_start = {pygame.K_LEFT}  # a key held since level start filters a player's input
    ship = game.ship
    angle, x, y = ship.angle, ship.x, ship.y
    game.update(1.0)
    assert (ship.angle - angle) % 360 == pytest.approx(ship.current_rotation_speed % 360)
    assert math.hypot(ship.x - x, ship.y - y) > 0


def test_a_piloted_ship_fires_on_the_game_clock(game):
    game.ship.weapon = None  # the plain gun, which cools down on the clock
    game.ship_pilot = Pilot(ShipCommand(turn=0, thrust=False, fire=True))
    game.update(0.0)
    game.update(0.0)
    assert len(game.projectiles) == 1  # frozen clock: still cooling down
    game.fake_now += 10.0
    game.update(0.0)
    assert len(game.projectiles) == 2


def test_a_piloted_ship_reaching_the_open_exit_completes_the_level_in_demo(game):
    game.ship_pilot = Pilot(ShipCommand(turn=0, thrust=False, fire=False))
    game.exit_opened = True
    game.maze.exit.set_activated(True, game.sound_manager)
    game.ship.x, game.ship.y = game.maze.exit.get_pos()
    game.update(1.0)
    assert game.exit_explosion_active


def test_a_piloted_ship_scores_out_in_demo(game):
    game.ship_pilot = Pilot(ShipCommand(turn=0, thrust=False, fire=False))
    game.update(1.0)
    assert not game.game_frozen
    game.fake_now += 100000.0
    game.update(1.0)
    assert game.game_frozen


def test_a_scored_out_game_ends_on_the_game_clock(game):
    game.ship_pilot = Pilot(ShipCommand(turn=0, thrust=False, fire=False))
    game.fake_now += 100000.0
    game.update(1.0)
    assert game.game_frozen
    started = game.fake_now
    duration = (game.game_over_death_duration + game.game_over_fade_duration
                + game.game_over_text_duration)
    game.update(1.0)
    assert game.game_over_active  # real time passing is nothing; the game clock has not moved
    assert game.levels_ended == []
    game.fake_now = started + duration + 1.0
    game.update(1.0)
    assert game.levels_ended == ['scored_out']
