"""The Campaign Hunter's senses: the hunter's own, then where the exit is and where an Ally beacon lies."""
import math
from types import SimpleNamespace

import numpy as np

import config
from entities.jev_beacon import JevBeacon
from hunter.campaign_senses import (CAMPAIGN_FEATURE_NAMES, CAMPAIGN_FEATURE_NAMES_V1, HELD_WEAPON_IDS, CampaignSenses,
                                   CampaignSurroundings, ShipBody)
from hunter.model import HunterSettings
from hunter.senses import FEATURE_NAMES, OBSERVATION_VERSION
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


def test_banked_kill_bonuses_do_not_lift_health_past_full(game):  # noqa: F811
    settings = HunterSettings()
    body = ShipBody(game.ship, game, settings)
    game.scoring.calculate_current_potential_score = lambda *a, **k: {'potential_score': 250.0}
    assert body.health == settings.health
    game.scoring.calculate_current_potential_score = lambda *a, **k: {'potential_score': 40.0}
    assert 0 < body.health < settings.health


def test_the_campaign_vector_of_a_live_level_is_finite(game):  # noqa: F811
    body = ShipBody(game.ship, game, HunterSettings())
    vector = CampaignSenses(HunterSettings()).perceive(body, CampaignSurroundings(game), IDLE, game.clock())
    assert np.isfinite(vector).all() and value(vector, 'exit_compass_present') == 1.0
    assert value(vector, 'fwd_speed') == 0.0 and value(vector, 'cooldown') == 0.0


def test_a_migrated_hunter_pilot_flies_the_same_with_campaign_senses(tmp_path):
    from hunter.neural.policy import NeuralPolicy
    from tools.migrate_policy import migrate
    migrate('assets/neural/hunter.npz', tmp_path / 'c.npz', CAMPAIGN_FEATURE_NAMES)
    hunter = NeuralPolicy.load('assets/neural/hunter.npz')
    campaign = NeuralPolicy.load(tmp_path / 'c.npz', CAMPAIGN_FEATURE_NAMES)
    rng = np.random.default_rng(1)
    senses = rng.normal(size=len(FEATURE_NAMES)).astype(np.float32)
    extra = np.full(len(CAMPAIGN_FEATURE_NAMES) - len(FEATURE_NAMES), 9.0, np.float32)
    a, _ = hunter.step(senses, hunter.initial_state())
    b, _ = campaign.step(np.concatenate([senses, extra]), campaign.initial_state())
    for head in a:
        assert np.allclose(a[head], b[head], atol=1e-5)


def test_a_held_weapon_part_way_through_its_reload_reads_as_that_share_of_its_reload(game):  # noqa: F811
    game.ship.equip_weapon('railgun')
    weapon = game.ship.weapon
    weapon.cooldown = weapon.definition.interval * 0.4      # frames, as the weapon counts them
    body = ShipBody(game.ship, game, HunterSettings())
    assert body.fire_remaining == weapon.cooldown / config.FPS
    vector = CampaignSenses(HunterSettings()).perceive(body, CampaignSurroundings(game), IDLE, game.clock())
    assert 0 < value(vector, 'cooldown') < 1
    assert math.isclose(value(vector, 'cooldown'), weapon.cooldown / weapon.definition.interval, rel_tol=1e-5)


def test_a_policy_file_loads_only_as_the_layout_it_was_built_for_and_migration_crosses(tmp_path):
    import pytest
    from hunter.neural.policy import InvalidPolicy, NeuralPolicy
    from tools.migrate_policy import migrate
    with pytest.raises(InvalidPolicy, match='layout differs'):
        NeuralPolicy.load('assets/neural/hunter.npz', CAMPAIGN_FEATURE_NAMES)
    migrate('assets/neural/hunter.npz', tmp_path / 'c.npz', CAMPAIGN_FEATURE_NAMES)
    with pytest.raises(InvalidPolicy, match='layout differs'):
        NeuralPolicy.load(tmp_path / 'c.npz')
    assert NeuralPolicy.load(tmp_path / 'c.npz', CAMPAIGN_FEATURE_NAMES).feature_names == CAMPAIGN_FEATURE_NAMES
    v1 = _v1_policy_file(tmp_path / 'v1.npz')                           # the first campaign layout: carried over
    assert NeuralPolicy.load(v1, CAMPAIGN_FEATURE_NAMES).feature_names == CAMPAIGN_FEATURE_NAMES
    with pytest.raises(InvalidPolicy, match='layout differs'):
        NeuralPolicy.load(v1)


def _live(game):  # noqa: F811  (a level, not the fixture)
    body = ShipBody(game.ship, game, HunterSettings())
    return CampaignSenses(HunterSettings()).perceive(body, CampaignSurroundings(game), IDLE, game.clock())


def _held(vector):
    return [w for w in HELD_WEAPON_IDS if value(vector, f'weapon_{w}') == 1.0]


def test_the_held_weapon_inputs_are_the_catalog_pickups_after_the_v1_layout():
    from weapons.catalog import CATALOG
    assert HELD_WEAPON_IDS == tuple(w for w in CATALOG.drops if CATALOG[w].behaviour != 'shield')
    assert CAMPAIGN_FEATURE_NAMES[:len(CAMPAIGN_FEATURE_NAMES_V1)] == CAMPAIGN_FEATURE_NAMES_V1
    assert CAMPAIGN_FEATURE_NAMES[len(CAMPAIGN_FEATURE_NAMES_V1):] == (
        tuple(f'weapon_{w}' for w in HELD_WEAPON_IDS) + ('shield_remaining',))


def test_the_held_weapon_inputs_name_the_pickup_weapon_in_hand(game):  # noqa: F811
    assert _held(_live(game)) == []                                     # the plain gun
    for weapon_id in HELD_WEAPON_IDS:
        game.ship.weapon.reset()
        game.ship.equip_weapon(weapon_id)
        assert _held(_live(game)) == [weapon_id]
    game.ship.weapon.reset()
    game.ship.equip_weapon('railgun')
    game.ship.weapon.ammo = 1
    game.ship.weapon.try_fire((game.ship.x, game.ship.y), game.ship.angle)   # the last round
    assert _held(_live(game)) == []


def test_shield_remaining_is_the_share_of_the_pickup_shield_left(game):  # noqa: F811
    assert value(_live(game), 'shield_remaining') == 0.0
    game.ship.equip_weapon('shield')
    assert value(_live(game), 'shield_remaining') == 1.0
    assert _held(_live(game)) == []                                     # a shield is not a weapon in hand
    full = game.ship.powerup_shield_timer
    for _ in range(int(full * 0.75)):                                   # three quarters of its frames go by
        game.ship.update(1.0)
    assert math.isclose(value(_live(game), 'shield_remaining'), 1 - int(full * 0.75) / full, rel_tol=1e-4)
    for _ in range(int(full)):
        game.ship.update(1.0)
    assert value(_live(game), 'shield_remaining') == 0.0


def _v1_policy_file(path, seed=3):
    """A V1-layout campaign policy with every V1 input weighted."""
    from hunter.neural.policy import random_policy
    policy = random_policy(seed)
    rng = np.random.default_rng(seed)
    width = len(CAMPAIGN_FEATURE_NAMES_V1)
    params = dict(policy.params, embed_w=rng.normal(0, 0.3, (policy.params['embed_w'].shape[0], width)).astype(np.float32))
    np.savez(path, mean=rng.normal(size=width).astype(np.float32), scale=rng.uniform(0.5, 2, width).astype(np.float32),
             feature_names=np.array(CAMPAIGN_FEATURE_NAMES_V1), observation_version=OBSERVATION_VERSION, **params)
    return path


def test_a_v1_campaign_pilot_flies_the_same_whatever_the_new_inputs_read(tmp_path):
    from hunter.neural.policy import NeuralPolicy
    path = _v1_policy_file(tmp_path / 'v1.npz')
    v1 = NeuralPolicy.load(path, CAMPAIGN_FEATURE_NAMES_V1)
    v2 = NeuralPolicy.load(path, CAMPAIGN_FEATURE_NAMES)
    assert v2.feature_names == CAMPAIGN_FEATURE_NAMES
    rng = np.random.default_rng(2)
    a_state, b_state = v1.initial_state(), v2.initial_state()
    for _ in range(5):
        senses = rng.normal(size=len(CAMPAIGN_FEATURE_NAMES_V1)).astype(np.float32)
        extra = rng.normal(0, 50, len(CAMPAIGN_FEATURE_NAMES) - len(senses)).astype(np.float32)
        a, a_state = v1.step(senses, a_state)
        b, b_state = v2.step(np.concatenate([senses, extra]), b_state)
        for head in a:
            assert np.allclose(a[head], b[head], atol=1e-5)


def test_only_the_known_v1_campaign_layout_is_carried_over(tmp_path):
    import pytest
    from hunter.neural.policy import InvalidPolicy, NeuralPolicy
    path = _v1_policy_file(tmp_path / 'v1.npz')
    with pytest.raises(InvalidPolicy, match='layout differs'):
        NeuralPolicy.load(path)                                          # not a hunter policy
    data = dict(np.load(path))
    cut = len(CAMPAIGN_FEATURE_NAMES_V1) - 1                             # some other prefix: refused
    np.savez(tmp_path / 'short.npz', **dict(data, feature_names=np.array(CAMPAIGN_FEATURE_NAMES_V1[:cut]),
                                            embed_w=data['embed_w'][:, :cut], mean=data['mean'][:cut],
                                            scale=data['scale'][:cut]))
    with pytest.raises(InvalidPolicy, match='layout differs'):
        NeuralPolicy.load(tmp_path / 'short.npz', CAMPAIGN_FEATURE_NAMES)


def test_training_from_a_v1_start_writes_a_v2_start(tmp_path):
    from hunter.neural.policy import NeuralPolicy
    from tools.ppo_neural import build_net, export, start_policy
    policy = start_policy(_v1_policy_file(tmp_path / 'v1.npz'), CAMPAIGN_FEATURE_NAMES)
    export(build_net(policy), policy, tmp_path / 'iter_start.npz')
    names = tuple(str(n) for n in np.load(tmp_path / 'iter_start.npz')['feature_names'])
    assert names == CAMPAIGN_FEATURE_NAMES
    reloaded = NeuralPolicy.load(tmp_path / 'iter_start.npz', CAMPAIGN_FEATURE_NAMES)
    assert not reloaded.params['embed_w'][:, len(CAMPAIGN_FEATURE_NAMES_V1):].any()
