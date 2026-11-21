import time
from unittest.mock import Mock

import pygame
import pytest

import config
from config import SETTINGS
from entities.projectile import Projectile
from hunter.duel import DuelResult
from hunter.neural import loader
from hunter.neural.policy import random_policy
from rendering.duel_view import banner_text
from rendering.fun_menu import FUN_ENTRIES
from tests.test_replay_loop import game, key  # noqa: F401 - pytest fixture


def to_fun(game):
    game.state = config.STATE_MENU
    game.main_menu.menu_selected_index = game.main_menu.menu_options.index('FUN')
    key(game, pygame.K_RETURN)


@pytest.fixture
def policy_file(tmp_path, monkeypatch):
    path = tmp_path / 'vibes.npz'
    random_policy(2).save(path)
    monkeypatch.setattr(loader, 'DEFAULT_PATH', str(path))
    return path


def test_fun_sits_between_how_to_play_and_select_profile(game):
    options = game.main_menu.menu_options
    assert options.index('FUN') == options.index('HOW TO PLAY') + 1 == options.index('SELECT PROFILE') - 1


def test_fun_opens_the_list_with_the_pilot_fight_and_cancel_goes_back(game):
    to_fun(game)
    assert game.state == config.STATE_FUN
    assert [e.title for e in game.fun_menu.entries] == ['Pilot fight: Rules vs Vibes', 'Flockers vs Wallcrawlers',
                                                         'Demo']
    assert game.fun_menu.selected_entry.key == 'pilot_duel'
    key(game, pygame.K_DOWN)
    key(game, pygame.K_DOWN)
    assert game.fun_menu.selected_entry.key == 'demo'
    key(game, pygame.K_DOWN)
    assert game.fun_menu.selected_index == 2                      # stop at the last entry
    key(game, pygame.K_ESCAPE)
    assert game.state == config.STATE_MENU


def test_the_list_is_data_so_more_levels_are_one_line(game):
    assert FUN_ENTRIES[0].key == 'pilot_duel'
    from rendering.fun_menu import FunEntry, FunMenu
    menu = FunMenu(game.screen, [FunEntry('A', 'a'), FunEntry('B', 'b')])
    menu.navigate(1)
    menu.navigate(1)
    assert menu.selected_index == 1 and menu.selected_entry.key == 'b'
    menu.navigate(-5)
    assert menu.selected_index == 0


def test_missing_neural_weights_keep_you_in_the_list_with_a_notice(game, monkeypatch, tmp_path):
    monkeypatch.setattr(loader, 'DEFAULT_PATH', str(tmp_path / 'absent.npz'))
    to_fun(game)
    key(game, pygame.K_RETURN)
    assert game.state == config.STATE_FUN and game.duel is None
    assert 'unavailable' in game.fun_menu.notice.lower()
    game.update(1.0)
    game.draw()                                                   # the notice draws without error


def test_starting_the_pilot_fight_builds_a_duel_and_leaves_the_normal_hunter_alone(game, policy_file):
    to_fun(game)
    key(game, pygame.K_RETURN)
    assert game.state == config.STATE_DUEL and game.duel is not None
    assert [s.name for s in game.duel.sides] == ['rules', 'vibes']
    assert game.hunter_flight is None and game.hunter_worker is None
    assert not game.duel.maze.exit.active                          # no exit portal drawn: nobody is flying out
    for _ in range(3):
        game.update(1.0)
    game.draw()
    game.end_duel()


def test_escape_leaves_the_duel_for_the_list_and_cleans_up(game, policy_file):
    to_fun(game)
    key(game, pygame.K_RETURN)
    worker = game.duel_worker
    worker.close = Mock(wraps=worker.close)
    key(game, pygame.K_ESCAPE)
    assert game.state == config.STATE_FUN and game.duel is None and game.duel_worker is None
    worker.close.assert_called_once()
    assert game.projectiles == [] and game.hunter is None


def test_a_finished_duel_shows_the_banner_then_returns_to_the_list(game, policy_file, monkeypatch):
    from experiments.harness import SyncWorker
    from hunter.rule_params import RULES
    monkeypatch.setattr(SETTINGS.game, 'duelTimeoutSeconds', 2.0)
    monkeypatch.setitem(type(game).__init__.__globals__, 'DUEL_BANNER_SECONDS', 0.0)   # `import game` finds a package, not game.py
    monkeypatch.setattr(type(game), '_make_duel_worker', lambda self: SyncWorker(RULES, time.monotonic))
    to_fun(game)
    key(game, pygame.K_RETURN)
    for _ in range(400):
        game.update(1.0)
        if game.state != config.STATE_DUEL:
            break
    assert game.state == config.STATE_FUN and game.duel is None


def test_the_banner_names_the_winner_or_the_draw():
    assert banner_text(DuelResult('rules', 'destroyed')) == 'RULES WINS'
    assert banner_text(DuelResult('vibes', 'destroyed')) == 'VIBES WINS'
    assert banner_text(DuelResult(None, 'draw')) == 'DRAW'
    assert banner_text(DuelResult(None, 'timeout')) == 'DRAW: TIME UP'


def test_the_duel_maze_is_a_complex_one_not_an_empty_box(game, policy_file):
    assert SETTINGS.game.duelMazeComplexity == 'COMPLEX' and SETTINGS.game.duelMazeGridSize == 26
    to_fun(game)
    key(game, pygame.K_RETURN)
    assert len(game.duel.maze.walls) > 200                     # an empty arena has only its outer wall
    for side in game.duel.sides:                                # and both hunters still start in the clear
        assert game.duel.maze.is_open(side.hunter.get_pos(), side.hunter.radius)
    game.end_duel()


def test_an_unknown_maze_complexity_leaves_you_in_the_list_with_a_notice(game, policy_file, monkeypatch):
    monkeypatch.setattr(SETTINGS.game, 'duelMazeComplexity', 'bogus')
    to_fun(game)
    key(game, pygame.K_RETURN)
    assert game.state == config.STATE_FUN and game.duel is None
    assert 'unavailable' in game.fun_menu.notice.lower()
