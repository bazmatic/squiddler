"""The Campaign Hunter's senses: the hunter's own, then where the exit is and where an Ally beacon lies."""
import math
from types import SimpleNamespace

import numpy as np

import config
from entities.jev_beacon import JevBeacon
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES, CampaignSenses, CampaignSurroundings, ShipBody
from hunter.model import HunterSettings
from hunter.senses import FEATURE_NAMES
from maze.wall_segment import WallSegment
from tests.hunter_fixture import HunterRig
from tests.test_campaign_pilot_seam import game  # noqa: F401  (fixture)
from tests.test_senses import IDLE, world, value as hunter_value

EXIT = (650.0, 150.0)


def value(vector, name):
    return float(vector[CAMPAIGN_FEATURE_NAMES.index(name)])


def level(walls=(), beacons=(), exit_opened=False):
    maze = world(walls)
    maze.exit = SimpleNamespace(get_pos=lambda: EXIT)
    return SimpleNamespace(maze=maze, roster=SimpleNamespace(all=lambda: []), projectiles=[], hunter=None,
                           powerup_crystals=list(beacons), exit_opened=exit_opened)


def perceive(game_like, body=None):
    body = body or HunterRig((350, 150))
    return CampaignSenses(HunterSettings()).perceive(body, CampaignSurroundings(game_like), IDLE, 0.0)


def beacon(x, y):
    b = JevBeacon((x, y))
    return b


def test_campaign_senses_are_the_hunter_senses_then_exit_and_beacon():
    assert CAMPAIGN_FEATURE_NAMES[:len(FEATURE_NAMES)] == FEATURE_NAMES
    vector = perceive(level())
    assert len(vector) == len(CAMPAIGN_FEATURE_NAMES) and vector.dtype == np.float32
    assert hunter_value(vector[:len(FEATURE_NAMES)], 'heading_cos') == 1.0


def test_the_exit_compass_follows_the_maze_route_not_the_straight_line():
    straight = perceive(level())
    assert value(straight, 'exit_compass_present') == 1.0
    assert abs(value(straight, 'exit_compass_sin')) < 1e-6        # nothing in the way: dead ahead
    walled = perceive(level(walls=[WallSegment((500, 0), (500, 800), 3)]))
    assert value(walled, 'exit_compass_sin') > 0.3                # the way round is down, through the gap
    assert value(walled, 'exit_compass_dist') > value(straight, 'exit_compass_dist')


def test_exit_open_reports_the_portal():
    assert value(perceive(level(exit_opened=False)), 'exit_open') == 0.0
    assert value(perceive(level(exit_opened=True)), 'exit_open') == 1.0


def test_the_beacon_compass_points_at_an_active_beacon_and_is_blank_without_one():
    blank = perceive(level())
    assert [value(blank, f'beacon_compass_{p}') for p in ('present', 'sin', 'cos', 'dist')] == [0.0] * 4
    wall = [WallSegment((500, 0), (500, 800), 3)]
    pointed = perceive(level(walls=wall, beacons=[beacon(350, 350)]))   # straight down, though no route is asked for
    assert value(pointed, 'beacon_compass_present') == 1.0
    assert math.isclose(value(pointed, 'beacon_compass_sin'), 1.0, abs_tol=1e-6)
    assert 0 < value(pointed, 'beacon_compass_dist') < 1
    gone = beacon(350, 350)
    gone.active = False
    assert value(perceive(level(beacons=[gone])), 'beacon_compass_present') == 0.0


def test_ship_health_is_the_remaining_potential_score(game):  # noqa: F811
    settings = HunterSettings()
    body = ShipBody(game.ship, game, settings)
    full = body.health
    assert 0 < full <= settings.health
    game.scoring.record_wall_collision()
    assert 0 < body.health < full


def test_the_campaign_vector_of_a_live_level_is_finite(game):  # noqa: F811
    body = ShipBody(game.ship, game, HunterSettings())
    vector = CampaignSenses(HunterSettings()).perceive(body, CampaignSurroundings(game), IDLE, game.clock())
    assert np.isfinite(vector).all() and value(vector, 'exit_compass_present') == 1.0
    assert value(vector, 'fwd_speed') == 0.0 and value(vector, 'cooldown') == 0.0
