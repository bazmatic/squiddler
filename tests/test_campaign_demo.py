"""Demo: the Campaign Hunter flies the player ship through each level, and Demo moves on when it leaves one."""
from unittest.mock import Mock

import pygame
import pytest

import config
import level_config
from config import SETTINGS
from entities.jev_beacon import JevBeacon
from game_handlers.autoplay import AutoplayDirector
from hunter.campaign_pilot import CampaignPilot
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES


class StubPolicy:
    """Always the same controls: (turn index, thrust, fire)."""

    def __init__(self, controls=(1, False, False)):
        self.controls = controls

    def initial_state(self):
        return None

    def act(self, senses, state, greedy=True, rng=None, **_):
        return self.controls, state


@pytest.fixture
def make_game(tmp_path, monkeypatch):
    from game import Game
    from profiles import ProfileManager
    pygame.init()
    screen = pygame.display.set_mode((1352, 878))
    monkeypatch.setattr(config, 'SCREEN_WIDTH', 1352)
    monkeypatch.setattr(config, 'SCREEN_HEIGHT', 878)
    monkeypatch.setattr(config, 'SPLASH_ENABLED', False)
    monkeypatch.delenv('START_LEVEL', raising=False)
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    monkeypatch.setattr(SETTINGS.game, 'hunterPilot', 'jev')
    monkeypatch.setattr(SETTINGS.autoplay, 'firstLevel', 4)
    monkeypatch.setattr(SETTINGS.autoplay, 'lastLevel', 6)
    monkeypatch.setattr(SETTINGS.autoplay, 'timeLimitSeconds', 90)
    monkeypatch.setattr(SETTINGS.autoplay, 'noKillSeconds', 0)
    manager = ProfileManager(tmp_path / 'profiles.json')
    monkeypatch.setitem(Game.__init__.__globals__, 'ProfileManager', lambda: manager)
    monkeypatch.setitem(Game.__init__.__globals__, 'SoundManager', Mock)
    monkeypatch.setattr(level_config, 'level_has_hunter', lambda _: False)
    games = []

    def make(autoplay=True, policy=StubPolicy()):
        if policy is not None:
            monkeypatch.setitem(Game.__init__.__globals__, 'load_campaign_policy', lambda: policy)
        if autoplay:
            monkeypatch.setenv('AUTOPLAY', '1')
        else:
            monkeypatch.delenv('AUTOPLAY', raising=False)
        game = Game(screen)
        games.append(game)
        return game
    make.profiles = tmp_path / 'profiles.json'
    yield make
    for game in games:
        game._close_hunter()
        if game.hunter_worker is not None:
            game.hunter_worker.close()
            game.hunter_worker.join(2)
    pygame.quit()


def profile_bytes(make_game):
    return make_game.profiles.read_bytes() if make_game.profiles.exists() else None


def put_ship_in_the_open_exit(game):
    game.exit_opened = True
    game.maze.exit.set_activated(True, game.sound_manager)
    game.ship.x, game.ship.y = game.maze.exit.get_pos()
    game.ship.vx = game.ship.vy = 0.0


def test_demo_is_off_by_default(make_game):
    game = make_game(autoplay=False)
    assert game.autoplay is None and game.ship_pilot is None
    assert game.state != config.STATE_PLAYING


def test_demo_flies_the_ship_even_before_a_campaign_policy_is_shipped(make_game, monkeypatch, tmp_path):
    import hunter.campaign_pilot
    monkeypatch.setattr(hunter.campaign_pilot, 'CAMPAIGN_POLICY_PATH', str(tmp_path / 'none.npz'))
    game = make_game(policy=None)
    assert isinstance(game.ship_pilot, CampaignPilot)
    assert game.ship_pilot.controller.policy.feature_names == CAMPAIGN_FEATURE_NAMES  # the hunter pilot, carried over
    ship = game.ship
    start = (ship.x, ship.y, ship.angle)
    for _ in range(120):
        game.update(1.0)
    assert (ship.x, ship.y, ship.angle) != pytest.approx(start, abs=1.0)


def test_demo_flies_the_ship_with_the_campaign_pilot(make_game):
    game = make_game(policy=StubPolicy((2, True, False)))   # turn right, thrust
    assert isinstance(game.ship_pilot, CampaignPilot)
    assert game.state == config.STATE_PLAYING and game.level == 4
    assert game.hunter is None
    ship = game.ship
    start = (ship.x, ship.y, ship.angle)
    for _ in range(30):
        game.update(1.0)
    assert (ship.x, ship.y) != pytest.approx(start[:2], abs=1.0)
    assert ship.angle != pytest.approx(start[2])
    ship.draw = Mock()
    game.draw()
    assert ship.draw.called


def test_fun_demo_starts_the_campaign_hunter(make_game):
    game = make_game(autoplay=False)
    game.state = config.STATE_FUN
    game.fun_menu.selected_index = next(
        i for i, entry in enumerate(game.fun_menu.entries) if entry.key == 'demo')
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=''))
    game.handle_events()
    assert game.autoplay is not None and isinstance(game.ship_pilot, CampaignPilot)
    assert game.state == config.STATE_PLAYING and game.level == 4
    assert game.player_has_moved


def test_leaving_demo_for_the_menu_hands_the_ship_back(make_game):
    game = make_game()
    game.state = config.STATE_QUIT_CONFIRM
    game.quit_confirmation_selection = 0
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=''))
    game.handle_events()
    assert game.state == config.STATE_MENU
    assert game.autoplay is None and game.ship_pilot is None


def test_reaching_the_exit_in_demo_moves_on_to_the_next_level_once(make_game):
    game = make_game()
    before = profile_bytes(make_game)
    put_ship_in_the_open_exit(game)
    game.update(1.0)
    assert game.exit_explosion_active
    # The time limit runs out while the exit explosion plays: the level still ends only once.
    expired = game.autoplay._started_at + 91
    game.autoplay.clock = lambda: expired if game.level == 4 else game.autoplay._started_at
    levels = []
    for _ in range(120):
        game.update(1.0)
        levels.append(game.level)
    assert levels[-1] == 5 and set(levels) == {4, 5}
    assert game.state == config.STATE_PLAYING and not game.exit_explosion_active
    assert profile_bytes(make_game) == before


def test_scoring_out_in_demo_moves_on_to_the_next_level(make_game):
    game = make_game()
    now = [1000.0]
    game.clock = lambda: now[0]
    real = game.scoring.calculate_current_potential_score
    game.scoring.calculate_current_potential_score = lambda *a, **k: (
        {'potential_score': 0, 'score_percentage': 0.0} if game.level == 4 else real(*a, **k))
    game.update(1.0)
    assert game.game_over_active
    now[0] += 60.0   # the power-down and GAME OVER have played
    game.update(1.0)
    assert game.level == 5
    assert game.state == config.STATE_PLAYING and not game.game_over_active


def test_demo_levels_never_touch_the_profile(make_game):
    game = make_game()
    before = profile_bytes(make_game)
    game.profile_manager = Mock(wraps=game.profile_manager)
    game.complete_level(success=True)
    game.complete_level(success=False)
    assert game.level == 6 and game.state == config.STATE_PLAYING
    game.profile_manager.record_level_result.assert_not_called()
    game.profile_manager.update_active_profile_progress.assert_not_called()
    assert profile_bytes(make_game) == before


def test_the_time_limit_advances_to_the_next_level(make_game):
    game = make_game()
    game.autoplay.clock = Mock(return_value=game.autoplay._started_at + 91)
    game.update(1.0)
    assert game.level == 5 and game.state == config.STATE_PLAYING


def test_the_ally_beacon_in_demo_summons_a_hunter_that_follows_the_ship(make_game):
    game = make_game()
    assert game.hunter is None
    beacon = JevBeacon((game.ship.x, game.ship.y))
    beacon.age = beacon.FORMATION_FRAMES   # formed: ready to be collected
    game.powerup_crystals.append(beacon)
    game.update(1.0)
    assert game.hunter is not None and game.hunter.active
    assert game.hunter_surroundings.friend() is game.ship
    game.hunter.active = False      # the ally is destroyed: the level goes on
    game.update(1.0)
    assert game.level == 4 and game.state == config.STATE_PLAYING


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_r_in_demo_restarts_the_level_as_demo_begins_one(make_game):
    game = make_game()
    clock = [1000.0]
    game.autoplay.clock = lambda: clock[0]
    game.autoplay.start_level(len(game.roster.alive()))
    game.ship_pilot.reset = Mock(wraps=game.ship_pilot.reset)
    clock[0] += 80                                    # most of the 90 s level timer used
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, unicode='r'))
    game.handle_events()
    assert game.level == 4 and game.state == config.STATE_PLAYING and game.player_has_moved
    game.ship_pilot.reset.assert_called_once()
    clock[0] += 80                                    # a fresh timer: 80 s into the restarted level
    game.update(1.0)
    assert game.level == 4


def test_the_no_kill_skip_waits_while_the_ship_heads_for_the_exit():
    clock = Clock()
    d = AutoplayDirector(4, 6, 300, clock=clock, no_kill_seconds=20)
    d.start_level(enemy_count=2)
    clock.now += 10
    assert d.end_reason(enemies_left=0) is None    # the last kill
    clock.now += 60
    assert d.end_reason(enemies_left=0) is None    # still on the way out
    clock.now += 231
    assert d.end_reason(enemies_left=0) == 'time'  # the safety net still holds
