"""A game has players, each with its own ship, driver, power clock and gun; with one player it plays as it always has."""
import pygame
import pytest

import config
from entities.player import Player
from tests.test_campaign_demo import make_game  # noqa: F401


def test_a_game_starts_with_one_player_whose_parts_are_the_games_ship_scoring_and_pilot(make_game):
    game = make_game(autoplay=False)
    assert len(game.players) == 1
    game.state = config.STATE_PLAYING
    game.start_level()
    player = game.players[0]
    assert game.ship is player.ship and game.scoring is player.scoring and game.ship_pilot is player.driver
    assert player.ship is not None and player.driver is None


def test_demos_pilot_is_the_first_players_driver(make_game):
    game = make_game()
    assert game.players[0].driver is game.ship_pilot is not None
    game.stop_autoplay()
    assert game.players[0].driver is None


def test_each_player_keeps_their_own_score_and_gun_and_first_move():
    a, b = Player(), Player()
    a.scoring.record_wall_collision()
    a.last_shot_time, a.has_moved = 500, True
    assert b.scoring.wall_collisions == 0 and b.last_shot_time == 0 and not b.has_moved


def test_the_level_wakes_when_any_player_has_moved_and_a_new_level_resets_everyone(make_game):
    game = make_game(autoplay=False)
    game.players.append(Player())
    assert not game.player_has_moved
    game.players[1].has_moved = True
    assert game.player_has_moved
    game.player_has_moved = False
    assert not any(p.has_moved for p in game.players)


def test_the_first_move_of_a_human_player_starts_the_level(make_game):
    game = make_game(autoplay=False)
    game.state = config.STATE_PLAYING
    game.start_level()
    game.level_start_time -= 10      # past the opening moment when input is ignored
    assert not game.player_has_moved
    pygame.event.clear()
    real = pygame.key.get_pressed
    pressed = {pygame.K_UP}
    try:
        pygame.key.get_pressed = lambda: type('K', (), {'__getitem__': lambda self, k: k in pressed})()
        game.update(1.0)
    finally:
        pygame.key.get_pressed = real
    assert game.players[0].has_moved and game.ship.game_started
