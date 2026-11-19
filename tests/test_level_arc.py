"""Campaign content remains valid, spawnable, and clear of the player start."""
import math
from unittest.mock import Mock

import pygame
import pytest

import config
import enemy_kinds
import level_config
import level_rules
from level_rules import anemones_that_fit

ARC = range(1, 25)


def enemies(level):
    layout = level_config.get_level_layout(level)
    if layout is not None:
        return layout.counts()
    return level_config.load_level_config(level)['enemies']


@pytest.mark.parametrize('level', range(1, 33))
def test_every_campaign_level_has_valid_enemy_counts_and_layout(level):
    data = level_config.load_level_config(level)
    assert set(data['enemies']) <= set(enemy_kinds.kinds())
    assert all(type(n) is int and n >= 0 for n in data['enemies'].values())
    assert 'layout' not in data or level_config.get_level_layout(level) is not None
    assert {k:v for k,v in enemies(level).items() if v} == {k:v for k,v in data['enemies'].items() if v}


@pytest.mark.parametrize('level', ARC)
def test_anemones_fit_the_maze(level):
    assert enemies(level).get('anemone', 0) <= anemones_that_fit(level_config.get_maze_grid_size(level))


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
    yield game
    game._close_hunter()


@pytest.mark.parametrize('level', list(ARC) + [25, 26, 30, 36, 42, 47, 66])
def test_every_enemy_the_level_asks_for_is_placed_clear_of_the_start(game, level):
    counts = level_config.get_level_enemy_counts(level) or level_rules.get_enemy_counts(level)
    asked_for = {
        'static': counts.static, 'patrol': counts.patrol, 'aggressive': counts.aggressive,
        'replay': counts.replay, 'flocker': counts.flocker, 'flighthouse': counts.flighthouse,
        'egg': counts.egg, 'anemone': level_config.get_level_anemone_count(level),
        'wallcrawler': level_config.get_level_wallcrawler_count(level),
        'split_boss': level_config.get_level_split_boss_count(level),
        'mother_boss': level_config.get_level_mother_boss_count(level)}
    game.level = level

    game.start_level()

    placed = {kind: len(game.roster.of_kind(kind)) for kind in asked_for}
    placed['aggressive'] += len(game.roster.of_kind('neural_jelly'))   # Jellies spawn as the neural kind
    assert placed == asked_for
    for enemy in game.roster.all():
        assert math.dist(enemy.get_pos(), game.maze.start_pos) >= config.ENEMY_START_CLEARANCE


ACTIVE = ('patrol', 'aggressive', 'flocker', 'wallcrawler', 'replay', 'belemnite', 'vibelet', 'neural_hunter', 'inchworm', 'sea_snake', 'mini_sea_snake', 'teeny_sea_snake')


def active_roster(level):
    counts = dict(enemies(level)) if level_config.get_level_layout(level) or level_config.load_level_config(level) else {}
    if not counts:
        generated = level_rules.get_enemy_counts(level)
        counts = {k: getattr(generated, k) for k in ('patrol', 'aggressive', 'flocker', 'replay')}
        counts['wallcrawler'] = level_config.get_level_wallcrawler_count(level)
    return {k: counts.get(k, 0) for k in ACTIVE if counts.get(k, 0)}


@pytest.mark.parametrize('level', [n for n in range(3, 80) if n not in (12, 18, 24, 25, 30, 32, 33, 35)
                                   and not level_rules.is_boss_level(n)])
def test_every_ordinary_level_is_built_around_one_active_type_in_numbers(level):
    roster = active_roster(level)
    assert roster, 'a level from the third on has something that moves'
    top, count = max(roster.items(), key=lambda kv: kv[1])
    assert count >= .5 * sum(roster.values()), f'no dominant type: {roster}'
    for kind in ('aggressive', 'flocker', 'wallcrawler'):
        assert roster.get(kind, 0) == 0 or roster[kind] >= 4, f'{kind} should come in numbers: {roster}'
