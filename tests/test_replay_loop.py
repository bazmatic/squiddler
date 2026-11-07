"""Replay and retry behavior through the real game and event handlers."""
import time
from unittest.mock import Mock

import pygame
import pytest

import config
import level_config
from game import Game
from profiles import ProfileManager
from maze.config import MazeComplexity


@pytest.fixture
def game(tmp_path, monkeypatch):
    pygame.init()
    screen = pygame.display.set_mode((1352, 878))
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)
    monkeypatch.setattr(config, 'SPLASH_ENABLED', False)
    monkeypatch.delenv('START_LEVEL', raising=False)
    manager = ProfileManager(tmp_path / 'profiles.json')
    manager.update_active_profile_progress(4, 400)
    monkeypatch.setitem(Game.__init__.__globals__, 'ProfileManager', lambda: manager)
    monkeypatch.setitem(Game.__init__.__globals__, 'SoundManager', Mock)
    monkeypatch.setattr(level_config, 'level_has_hunter', lambda _: False)
    monkeypatch.setattr(level_config, 'get_maze_complexity', lambda _: MazeComplexity.EMPTY)
    monkeypatch.setattr(level_config, 'get_maze_grid_size', lambda _: 5)
    game = Game(screen)
    yield game
    game._close_hunter()


def key(game, code):
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=code, unicode=''))
    game.handle_events()


def test_retry_during_play_and_power_out_restores_starting_total(game):
    game.state = config.STATE_PLAYING
    game.start_level()
    for power_out in (False, True):
        old_ship = game.ship
        game.scoring.total_score = 999
        game.game_over_active = game.game_frozen = power_out
        key(game, pygame.K_r)
        assert game.state == config.STATE_PLAYING
        assert game.ship is not old_ship
        assert game.scoring.total_score == 400
        assert not game.game_over_active and not game.game_frozen
        assert game.profile_manager.get_active_profile().bests == {}


@pytest.mark.parametrize('skip', ['key', 'button', 'timeout'])
def test_power_out_can_be_skipped_and_never_records_a_clear(game, monkeypatch, skip):
    game.state = config.STATE_PLAYING
    game.start_level()
    game.player_has_moved = True
    game.scoring.level_start_time = time.time() - 1000
    game.update(1)
    assert game.game_over_active
    game.sound_manager.play_power_down.assert_called_once()
    if skip == 'key':
        key(game, pygame.K_SPACE)
    elif skip == 'button':
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.JOYBUTTONDOWN, button=0))
        game.handle_events()
    else:
        monkeypatch.setattr(game, 'clock', lambda: game.game_over_start_time + game.game_over_death_duration + 1.15)
        game.update(1)
    assert game.state == config.STATE_LEVEL_COMPLETE
    assert not game.level_succeeded
    assert game.profile_manager.get_active_profile().bests == {}
    assert game.level_complete_menu.get_selected_option() == 'RETRY LEVEL'
    key(game, pygame.K_r)
    assert game.state == config.STATE_PLAYING and game.scoring.total_score == 400


@pytest.mark.parametrize("power_out", [False, True])
def test_controller_y_restarts_and_controls_explain_it(game, power_out):
    from rendering.settings_menu import CONTROL_GROUPS
    game.input_handler.controllers = [Mock(get_numbuttons=lambda: 12)]
    assert game.input_handler.is_controller_restart_pressed(3)
    assert not game.input_handler.is_controller_restart_pressed(2)
    game.state = config.STATE_PLAYING
    game.start_level()
    old_ship = game.ship
    game.game_over_active = game.game_frozen = power_out
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.JOYBUTTONDOWN, button=3))
    game.handle_events()
    assert game.ship is not old_ship
    groups = dict(CONTROL_GROUPS)
    assert ('R', 'Restart level') in groups['KEYBOARD']
    assert ('Y', 'Restart level') in groups['CONTROLLER']


def test_replay_records_bests_without_changing_progress_then_continues(game):
    game.state = config.STATE_PLAYING
    game.level = 4
    game.start_level()
    assert game.replaying
    game.complete_level()
    profile = game.profile_manager.get_active_profile()
    assert (profile.level, profile.total_score) == (5, 400)
    assert 4 in profile.bests and game.level_result.first_clear
    assert game.scoring.total_score == game.level_score_breakdown['total_score'] == 400
    key(game, pygame.K_RETURN)
    assert game.level == 5 and not game.replaying
    game.complete_level()
    assert profile.level == 6 and profile.total_score > 400
    assert 5 in profile.bests


@pytest.mark.parametrize('start_level,replaying', [(2, True), (5, False), (7, False)])
def test_start_level_override_records_every_clear(game, start_level, replaying):
    game.initial_start_level = start_level
    key(game, pygame.K_RETURN)
    assert game.level == start_level and game.replaying == replaying
    game.complete_level()
    assert start_level in game.profile_manager.get_active_profile().bests


def test_levels_menu_opens_returns_and_starts_selected_level(game):
    game.main_menu.menu_selected_index = game.main_menu.menu_options.index('LEVELS')
    key(game, pygame.K_RETURN)
    assert game.state == config.STATE_LEVEL_SELECT
    assert game.level_select_menu.selected_level == 5
    key(game, pygame.K_ESCAPE)
    assert game.state == config.STATE_MENU
    key(game, pygame.K_RETURN)
    key(game, pygame.K_LEFT)
    assert game.level_select_menu.selected_level == 4
    key(game, pygame.K_RETURN)
    assert game.level == 4 and game.replaying and game.state == config.STATE_PLAYING
    game.complete_level()
    key(game, pygame.K_RETURN)
    assert game.level == 5 and not game.replaying


def test_six_main_menu_buttons_fit_above_badge(game):
    game.main_menu.set_profile_info('Player1', 5)
    game.main_menu.draw()
    buttons = game.main_menu.menu_buttons
    assert len(buttons) == 6
    assert all(a.position[1] + a.height/2 < b.position[1] - b.height/2 for a,b in zip(buttons,buttons[1:]))
    last_bottom = buttons[-1].position[1] + buttons[-1].height / 2
    badge_top = game.screen.get_height() - game.main_menu._scaled(56) - game.main_menu.profile_badge.get_height()
    assert last_bottom < badge_top


def test_controller_grid_navigation_and_locked_tile(game):
    game.state = config.STATE_LEVEL_SELECT
    game.input_handler.controllers = [Mock(get_numbuttons=lambda: 12)]
    handler = game.state_handler_registry.get_handler(game.state)
    handler.handle_controller(pygame.event.Event(pygame.JOYHATMOTION, value=(-1, 0)), game)
    assert game.level_select_menu.selected_level == 4
    game.level_select_menu.selected_level = 6
    handler.handle_controller(pygame.event.Event(pygame.JOYBUTTONDOWN, button=0), game)
    assert game.state == config.STATE_LEVEL_SELECT
    handler.handle_controller(pygame.event.Event(pygame.JOYBUTTONDOWN, button=1), game)
    assert game.state == config.STATE_MENU


def test_retry_of_ordinary_clear_keeps_original_total_and_play_mode(game):
    game.state = config.STATE_PLAYING
    game.start_level()
    assert not game.replaying
    game.complete_level()
    key(game, pygame.K_r)
    assert game.scoring.total_score == 400
    assert not game.replaying
    game.complete_level()
    assert game.scoring.total_score > 400


def test_replay_continue_to_another_old_level_stays_a_replay(game):
    game.level = 2
    game.state = config.STATE_PLAYING
    game.start_level()
    game.complete_level()
    key(game, pygame.K_RETURN)
    assert game.level == 3 and game.replaying
    assert game.scoring.total_score == 400


def test_game_over_text_is_visible_as_soon_as_fade_finishes(game, monkeypatch):
    game.state = config.STATE_PLAYING
    game.start_level()
    game.player_has_moved = True
    game.scoring.level_start_time = time.time() - 1000
    game.update(1)
    monkeypatch.setattr(game, 'clock', lambda: game.game_over_start_time + game.game_over_death_duration + 0.51)
    game.draw()
    # The fully black fade is behind the white GAME OVER label.
    center = game.screen.subsurface(pygame.Rect(300, 300, 750, 260))
    white = pygame.mask.from_threshold(center, (255, 255, 255), (20, 20, 20, 255))
    assert white.count() > 500
    assert game.state == config.STATE_PLAYING


def test_retry_stops_audio_from_the_abandoned_run(game, monkeypatch):
    from sounds import SoundManager
    monkeypatch.setattr(config, 'SOUND_ENABLED', True)
    game.sound_manager = SoundManager()
    game.state = config.STATE_PLAYING
    game.start_level()
    sound = pygame.mixer.Sound(buffer=b'\x00\x00' * 4000)
    channel = sound.play(loops=-1)
    assert channel.get_busy()
    game.restart_level()
    assert not channel.get_busy()
