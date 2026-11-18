"""Campaign pursuit keeps a target without concealing other visible threats."""
from types import SimpleNamespace

import creatures
import numpy as np
import pytest

from hunter.campaign_senses import CampaignSenses
from hunter.model import HunterSettings
from hunter.senses import Senses, MAX_ENEMIES
from maze.wall_segment import WallSegment
from tests.hunter_fixture import HunterRig
from tests.test_senses import IDLE, world, value


def observe(senses, ship, maze, enemies, now=0):
    return senses.observe(ship, maze, enemies, [], [], None, IDLE, now)


@pytest.fixture
def scene():
    ship = HunterRig((350, 350))
    target = creatures.build('wallcrawler', (450, 350), 1)
    rival = creatures.build('patrol', (350, 650), 1)
    senses = CampaignSenses(HunterSettings(), sticky_targets=True)
    maze = world()
    observe(senses, ship, maze, [target, rival])
    assert senses.preferred_enemy is target
    return senses, ship, maze, target, rival


def test_closer_rival_does_not_replace_the_focus_or_compass(scene):
    senses, ship, maze, target, rival = scene
    rival.y = 380
    vector = observe(senses, ship, maze, [rival, target], 1)
    assert senses.preferred_enemy is target
    assert value(vector, 'enemy0_sin') == pytest.approx(0, abs=1e-6)
    assert value(vector, 'enemy1_sin') == pytest.approx(1, abs=1e-6)
    assert value(vector, 'enemy_compass_sin') == pytest.approx(0, abs=1e-6)


def test_knocked_away_wounded_target_is_kept(scene):
    senses, ship, maze, target, rival = scene
    target.x, target.hit_points = 650, 1
    rival.y = 450
    observe(senses, ship, maze, [target, rival], 1)
    assert senses.preferred_enemy is target


def test_hidden_target_keeps_compass_but_never_becomes_a_visible_contact(scene):
    senses, ship, _, target, rival = scene
    maze = world([WallSegment((400, 0), (400, 800), 3)])
    rival.y = 450
    vector = observe(senses, ship, maze, [target, rival], 1)
    assert senses.preferred_enemy is target
    assert value(vector, 'enemy0_sin') == pytest.approx(1, abs=1e-6)
    assert value(vector, 'enemy1_present') == 0
    assert value(vector, 'enemy_compass_present') == 1


def test_focus_is_not_dropped_when_more_than_six_enemies_are_closer(scene):
    senses, ship, maze, target, _ = scene
    target.x = 650
    others = [creatures.build('patrol', (360 + i * 5, 400), 1) for i in range(MAX_ENEMIES + 4)]
    vector = observe(senses, ship, maze, others + [target], .1)
    assert senses.preferred_enemy is target
    assert value(vector, 'enemy0_cos') == pytest.approx(1, abs=1e-6)
    assert value(vector, 'enemy0_dist') == pytest.approx(300 / senses.settings.sensor_range(maze))
    assert sum(value(vector, f'enemy{i}_present') for i in range(MAX_ENEMIES)) == MAX_ENEMIES


def test_death_removal_and_reset_release_focus(scene):
    senses, ship, maze, target, rival = scene
    target.active = False
    observe(senses, ship, maze, [target, rival], .1)
    assert senses.preferred_enemy is rival
    observe(senses, ship, maze, [], .2)
    assert senses.preferred_enemy is None
    target.active = True
    observe(senses, ship, maze, [target, rival], .3)
    senses.reset()
    assert senses.preferred_enemy is None
    rival.y = 360
    observe(senses, ship, maze, [target, rival], .4)
    assert senses.preferred_enemy is rival


def test_unreachable_target_releases_immediately(scene):
    senses, ship, _, target, rival = scene
    maze = world([WallSegment((400, 0), (400, 1000), 3)])
    observe(senses, ship, maze, [target, rival], .1)
    assert senses.preferred_enemy is rival


def test_much_worse_chase_requires_two_continuous_seconds(scene):
    senses, ship, maze, target, rival = scene
    target.x, target.y = 950, 950
    rival.y = 360
    observe(senses, ship, maze, [target, rival], 1)
    observe(senses, ship, maze, [target, rival], 2.9)
    assert senses.preferred_enemy is target
    # Recovery cancels the timer, so a second brief disadvantage does not end the chase.
    target.x, target.y = 450, 350
    observe(senses, ship, maze, [target, rival], 3)
    target.x, target.y = 950, 950
    observe(senses, ship, maze, [target, rival], 4)
    observe(senses, ship, maze, [target, rival], 5.9)
    assert senses.preferred_enemy is target
    observe(senses, ship, maze, [target, rival], 6)
    assert senses.preferred_enemy is rival


def test_disabled_focus_retains_exact_base_observations():
    ship, maze = HunterRig((350, 350)), world()
    target = creatures.build('wallcrawler', (450, 350), 1)
    rival = creatures.build('patrol', (350, 650), 1)
    campaign, base = CampaignSenses(HunterSettings()), Senses(HunterSettings())
    for now in (0, 1, 2):
        a = observe(campaign, ship, maze, [target, rival], now)
        b = observe(base, ship, maze, [target, rival], now)
        np.testing.assert_array_equal(a, b)
        rival.y -= 140
    assert campaign.preferred_enemy is None


def test_pursuit_reward_uses_the_pilots_focus_and_target_switch_cannot_pay(scene):
    from experiments.campaign_rollout import _Recorder
    from hunter.campaign_layout import CAMPAIGN_FEATURE_NAMES
    senses, ship, maze, target, rival = scene
    # Move west toward the locked target while the nearer rival is east.
    target.x, rival.x, rival.y = 150, 450, 350
    pilot = SimpleNamespace(target_enemy=target, on_observed=lambda listener: None)
    game = SimpleNamespace(ship=ship, maze=maze, roster=SimpleNamespace(alive=lambda: [target, rival]))
    recorder = _Recorder(pilot, game)
    paid = []
    recorder.add = lambda event, amount: paid.append((event, amount))
    vector = np.zeros(len(CAMPAIGN_FEATURE_NAMES), dtype=np.float32)
    recorder._pay_progress(vector)
    assert recorder._progress[1] is target
    ship.x -= 20
    recorder._pay_progress(vector)
    assert sum(amount for event, amount in paid if event == 'hunt_progress') > 0
    paid.clear()
    pilot.target_enemy = rival
    recorder._pay_progress(vector)
    assert not paid
    assert recorder._progress[1] is rival


@pytest.fixture
def recovery():
    ship, maze = HunterRig((350, 350)), world()
    target = creatures.build('wallcrawler', (650, 350), 1)
    rival = creatures.build('patrol', (350, 850), 1)
    senses = CampaignSenses(HunterSettings(), sticky_targets=True, target_recovery=True)
    observe(senses, ship, maze, [target, rival], 0)
    rival.y = 450
    return senses, ship, maze, target, rival


def test_recovery_requires_sticky_targeting():
    with pytest.raises(ValueError, match='requires sticky'):
        CampaignSenses(HunterSettings(), target_recovery=True)


def test_stalled_chase_switches_at_four_seconds(recovery):
    senses, ship, maze, target, rival = recovery
    observe(senses, ship, maze, [target, rival], 3.99)
    assert senses.preferred_enemy is target
    observe(senses, ship, maze, [target, rival], 4)
    assert senses.preferred_enemy is rival
    assert senses.target_switch_reason == 'stalled'
    assert senses._target_cooldowns[target] == 10


def test_small_improvements_accumulate_and_reset_stall_timer(recovery):
    senses, ship, maze, target, rival = recovery
    cell = max(maze.cell_size_x, maze.cell_size_y)
    for now in (1, 2, 3):
        ship.x += cell * .1
        observe(senses, ship, maze, [target, rival], now)
    assert senses._target_progress_at == 3
    observe(senses, ship, maze, [target, rival], 6.9)
    assert senses.preferred_enemy is target


def test_damage_resets_progress_even_after_knockback(recovery):
    senses, ship, maze, target, rival = recovery
    target.hit_points -= 1
    target.x += 100
    observe(senses, ship, maze, [target, rival], 3.9)
    assert senses._target_progress_at == 3.9
    assert senses._target_progress_cost == senses._route_cost((ship.x, ship.y), maze, target)
    observe(senses, ship, maze, [target, rival], 7.8)
    assert senses.preferred_enemy is target


@pytest.mark.parametrize('alternative', ['hidden', 'too_far', 'unreachable'])
def test_recovery_rejects_unsuitable_alternatives(recovery, alternative, monkeypatch):
    senses, ship, maze, target, rival = recovery
    if alternative == 'hidden':
        maze = world([WallSegment((0, 400), (800, 400), 3)])
    elif alternative == 'too_far':
        rival.y = 620
    else:
        maze = world([WallSegment((0, 400), (1000, 400), 3)])
    observe(senses, ship, maze, [target, rival], 4)
    assert senses.preferred_enemy is target


def test_cooldown_blocks_switch_back_then_expires(recovery):
    senses, ship, maze, target, rival = recovery
    observe(senses, ship, maze, [target, rival], 4)
    target.x, rival.y = 400, 650
    observe(senses, ship, maze, [target, rival], 8)
    assert senses.preferred_enemy is rival
    observe(senses, ship, maze, [target, rival], 10)
    assert senses.preferred_enemy is target
    assert target not in senses._target_cooldowns


@pytest.mark.parametrize('reason', ['dead', 'removed', 'unreachable'])
def test_forced_switch_overrides_commitment_and_cooldown(recovery, reason, monkeypatch):
    senses, ship, maze, target, rival = recovery
    observe(senses, ship, maze, [target, rival], 4)
    enemies = [target, rival]
    if reason == 'dead':
        rival.active = False
    elif reason == 'removed':
        enemies = [target]
    else:
        maze = world([WallSegment((0, 400), (1000, 400), 3)])
    observe(senses, ship, maze, enemies, 4.1)
    assert senses.preferred_enemy is target
    assert senses.target_switch_reason == 'unavailable'


def test_single_enemy_and_reset_clear_recovery_state(recovery):
    senses, ship, maze, target, rival = recovery
    observe(senses, ship, maze, [target], 20)
    assert senses.preferred_enemy is target
    observe(senses, ship, maze, [target, rival], 21)
    assert senses.preferred_enemy is rival
    senses.reset()
    assert senses.preferred_enemy is None
    assert not senses._target_cooldowns
    assert senses._target_hits is None
    assert senses._target_disadvantaged_since is None
    observe(senses, ship, maze, [], 0)
    assert senses.preferred_enemy is None


def test_disabled_recovery_is_identical_to_existing_persistence():
    ship, maze = HunterRig((350, 350)), world()
    enemies = [creatures.build('wallcrawler', (650, 350), 1), creatures.build('patrol', (350, 850), 1)]
    a = CampaignSenses(HunterSettings(), sticky_targets=True)
    b = CampaignSenses(HunterSettings(), sticky_targets=True, target_recovery=False)
    for now in (0, 1, 4, 10):
        np.testing.assert_array_equal(observe(a, ship, maze, enemies, now), observe(b, ship, maze, enemies, now))
        assert a.preferred_enemy is b.preferred_enemy
        enemies[1].y = 400


def test_legacy_disadvantage_waits_for_commitment_and_sets_cooldown(recovery):
    senses, ship, maze, target, rival = recovery
    target.x, target.y, rival.y = 950, 950, 360
    observe(senses, ship, maze, [target, rival], 1)
    assert senses._target_disadvantaged_since is None
    observe(senses, ship, maze, [target, rival], 2)
    observe(senses, ship, maze, [target, rival], 3.9)
    assert senses.preferred_enemy is target
    observe(senses, ship, maze, [target, rival], 4)
    assert senses.preferred_enemy is rival
    assert senses.target_switch_reason == 'disadvantaged'
    assert senses._target_cooldowns[target] == 10


def test_recovery_switch_does_not_generate_progress_reward(recovery):
    from experiments.campaign_rollout import _Recorder
    from hunter.campaign_layout import CAMPAIGN_FEATURE_NAMES
    senses, ship, maze, target, rival = recovery
    pilot = SimpleNamespace(target_enemy=target, on_observed=lambda listener: None)
    game = SimpleNamespace(ship=ship, maze=maze, roster=SimpleNamespace(alive=lambda: [target, rival]))
    recorder = _Recorder(pilot, game)
    paid = []
    recorder.add = lambda event, amount: paid.append((event, amount))
    vector = np.zeros(len(CAMPAIGN_FEATURE_NAMES), dtype=np.float32)
    recorder._pay_progress(vector)
    observe(senses, ship, maze, [target, rival], 4)
    assert senses.preferred_enemy is rival
    pilot.target_enemy = senses.preferred_enemy
    recorder._pay_progress(vector)
    assert not paid
    ship.y += 20
    recorder._pay_progress(vector)
    assert sum(amount for event, amount in paid if event == 'hunt_progress') > 0


def test_brief_occlusion_does_not_release_recovery_target(recovery):
    senses, ship, _, target, rival = recovery
    maze = world([WallSegment((500, 0), (500, 800), 3)])
    observe(senses, ship, maze, [target, rival], 1)
    assert senses.preferred_enemy is target
    observe(senses, ship, world(), [target, rival], 2)
    assert senses.preferred_enemy is target


def test_equal_cost_recovery_candidates_keep_roster_order(recovery):
    senses, ship, maze, target, rival = recovery
    other = creatures.build('patrol', (rival.x, rival.y), 1)
    observe(senses, ship, maze, [target, other, rival], 4)
    assert senses.preferred_enemy is other


def test_visible_target_in_curved_boundary_cell_keeps_commitment():
    # The curved rock occupies less than its coarse border cell. A crab can
    # enter the open part of that cell without crossing any actual wall.
    ship = HunterRig((450, 250))
    maze = world([WallSegment((0, 20), (1000, 20), 3)])
    maze.blocks = {(4, 0): 3}
    target = creatures.build('patrol', (450, 150), 1)
    rival = creatures.build('patrol', (750, 250), 1)
    senses = CampaignSenses(HunterSettings(), sticky_targets=True, target_recovery=True)
    observe(senses, ship, maze, [target, rival], 0)
    assert senses.preferred_enemy is target
    target.y = 50
    assert senses._routes.cheapest_route(maze, (ship.x, ship.y), [(target.x, target.y)],
                                         senses.hit_point_cells) is None
    observe(senses, ship, maze, [target, rival], .1)
    assert senses.preferred_enemy is target
    assert senses._route_cost((ship.x, ship.y), maze, target) == pytest.approx(200)


@pytest.mark.parametrize('gate', [False, True])
def test_boundary_fallback_does_not_see_through_actual_walls_or_gates(gate):
    ship = HunterRig((450, 250))
    barrier = WallSegment((0, 100), (1000, 100), 3)
    maze = world([] if gate else [barrier])
    if gate:
        maze.gate_segments = [barrier]
    maze.blocks = {(4, 0): 3}
    target = creatures.build('patrol', (450, 50), 1)
    senses = CampaignSenses(HunterSettings(), sticky_targets=True, target_recovery=True)
    assert senses._route_cost((ship.x, ship.y), maze, target) == float('inf')


def test_unreachable_interior_cell_is_not_reclassified_as_directly_reachable():
    ship = HunterRig((450, 250))
    maze = world()
    # Even if a centre ray is clear, do not bypass the grid's interior
    # traversability rules: this fallback is specifically for the outer rim.
    maze.blocks = {(4, 4): 3}
    maze.strong_cells = {(4, 4)}
    target = creatures.build('patrol', (450, 450), 1)
    senses = CampaignSenses(HunterSettings(), sticky_targets=True, target_recovery=True)
    assert senses._route_cost((ship.x, ship.y), maze, target) == float('inf')


def test_boundary_fallback_requires_target_within_sensor_range():
    ship = HunterRig((450, 950))
    maze = world()
    maze.blocks = {(4, 0): 3}
    target = creatures.build('patrol', (450, 50), 1)
    senses = CampaignSenses(HunterSettings(), sticky_targets=True, target_recovery=True)
    assert senses._route_cost((ship.x, ship.y), maze, target) == float('inf')
