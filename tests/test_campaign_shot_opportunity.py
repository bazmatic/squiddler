"""Missed-opportunity credit uses real sensed targets and ballistic collision geometry, never a loose aim cone."""
import math

import numpy as np
import pytest

from hunter.campaign_senses import CampaignSenses, CampaignSurroundings, ShipBody
from hunter.settings_for import hunter_settings_for
from learning.campaign_shot_opportunity import opportunity
from learning.campaign_shot_opportunity import aim_feedback, rough_opportunity, observed_targets, _lanes
from maze.wall_segment import WallSegment
from tests.test_campaign_pilot_seam import game  # noqa: F401


def open_scene(game):
    maze, ship = game.maze, game.ship
    maze.curved = False
    for y, row in enumerate(maze.grid):
        for x in range(len(row)):
            row[x] = int(x in (0, maze.grid_width-1) or y in (0, maze.grid_height-1))
    maze.rebuild_walls()
    game._close_enemy_hunters()
    game.encounter.close()
    game.roster.clear()
    ship.x, ship.y = maze.position_calculator.grid_center_to_screen(2, 3)
    ship.prev_x, ship.prev_y = ship.x, ship.y
    ship.vx = ship.vy = 0
    ship.angle = 0
    ship.shield_active = False
    ship.shield_initial_timer = 0
    ship.weapon.reset()
    game.__dict__.pop('last_shot_time', None)
    target = game.encounter.spawn('static', (ship.x + 2 * maze.cell_size_x, ship.y), 1)
    game.encounter.ready()
    target.vx = target.vy = 0
    return target


def sensed(game):
    settings = hunter_settings_for('neural')
    return CampaignSenses(settings).perceive(ShipBody(game.ship, game, settings), CampaignSurroundings(game),
                                             (1, 0, 0), 0)


def test_ready_direct_shot_is_an_opportunity_but_a_loose_twenty_degree_near_miss_is_not(game):
    target = open_scene(game)
    assert opportunity(game, sensed(game))[0] == (target,)
    target.y += 40
    assert abs(math.degrees(math.atan2(target.y-game.ship.y, target.x-game.ship.x))) < 20
    assert opportunity(game, sensed(game))[0] == ()


def test_graded_rule_accepts_rough_aim_and_rewards_increasing_intercept_accuracy(game):
    target = open_scene(game)
    scores = []
    muzzle = game.ship.x + game.ship.radius + 5
    for degrees in (0, 10, 20, 30, 45):
        angle = math.radians(degrees)
        target.x, target.y = muzzle + 180*math.cos(angle), game.ship.y + 180*math.sin(angle)
        observations = sensed(game)
        feedback = aim_feedback(game, observed_targets(game, observations), tuple(_lanes(game.ship, 0)), 0)
        scores.append(feedback.accuracy)
        assert feedback.rough == (degrees <= 30)
        assert bool(rough_opportunity(game, observations)[0]) == (degrees <= 30)
    assert scores[0] == pytest.approx(1)
    assert scores[0] > scores[1] > scores[2] > scores[3]
    assert scores[3] == pytest.approx(0, abs=1e-6)
    assert scores[4] == 0


def test_graded_volley_feedback_waives_rough_shot_costs_and_preserves_historical_rules():
    from experiments.campaign_rollout import fly_campaign
    from experiments.student import StudentSpec
    from hunter.campaign_pilot import CampaignPilot
    from learning.campaign_rewards import TABLE, CampaignShaping

    class Controls:
        def __init__(self, fire):
            self.fire = fire

        def initial_state(self):
            return None

        def act(self, senses, state, *args, **kwargs):
            return (1, False, self.fire), None

    class Practice(CampaignPilot):
        def __init__(self, degrees, fire=True):
            super().__init__(Controls(fire), 6)
            self.degrees = degrees

        def decide(self, game):
            if self.controller.frames == 0:
                target = open_scene(game)
                angle = math.radians(self.degrees)
                target.x = game.ship.x + game.ship.radius + 5 + 180*math.cos(angle)
                target.y = game.ship.y + 180*math.sin(angle)
                self.ship = game.ship
            return super().decide(game)

    values = {k: 0 for k in TABLE.defaults}
    values.update(shot=-.1, unaimed_shot=-.6, missed_opportunity=-.2, shot_accuracy=.05, enemy_hit=4)
    shaping = CampaignShaping(**values)
    def fly(degrees, graded=True, fire=True, power=2):
        pilot = Practice(degrees, fire)
        episode = fly_campaign(1, 300, StudentSpec('unused'), horizon=.5, pilot=pilot, shaping=shaping,
                                graded_shooting=graded, accuracy_power=power)
        assert 'fire' not in pilot.ship.__dict__  # temporary volley observer was removed
        return episode
    perfect, rough, outside = fly(0), fly(20), fly(45)
    for episode in (perfect, rough):
        assert episode['stats'].get('shots', 0) > 0
        assert episode['reward_totals']['shot'] == episode['reward_totals']['unaimed_shot'] == 0
    assert perfect['rewards'].sum() > rough['rewards'].sum() > 0
    assert perfect['stats'].get('damaging_hits', 0) > 0
    assert perfect['reward_totals']['enemy_hit'] == 4 * perfect['stats'].get('damaging_hits', 0)
    maximum_aim_only = perfect['reward_totals']['shot_accuracy'] / perfect['stats'].get('shots', 0)
    assert maximum_aim_only == pytest.approx(.05)
    assert 5 * maximum_aim_only < shaping.enemy_hit * .1  # even five best aim-only credits stay below 10% of a hit
    historical_linear = fly(20, power=1)
    assert 0 < rough['reward_totals']['shot_accuracy'] < historical_linear['reward_totals']['shot_accuracy']
    np.testing.assert_array_equal(rough['controls'], historical_linear['controls'])
    assert outside['reward_totals']['shot'] < 0 and outside['reward_totals']['unaimed_shot'] < 0
    assert outside['reward_totals']['shot_accuracy'] == 0
    waiting = fly(20, fire=False)
    assert waiting['reward_totals']['missed_opportunity'] < 0
    assert waiting['reward_totals']['shot_accuracy'] == 0
    legacy = fly(20, graded=False)
    assert legacy['reward_totals']['shot'] < 0
    assert legacy['reward_totals']['shot_accuracy'] == 0


def test_target_motion_can_remove_or_create_a_real_intercept(game):
    target = open_scene(game)
    target.vy = 6
    assert opportunity(game, sensed(game))[0] == ()
    target.y += 40
    # Lead a crossing target: it reaches the nose's firing line when the projectile arrives.
    speed = game.ship.weapon.definition.base_speed * game.ship.weapon.definition.speed
    time = (target.x - game.ship.x - game.ship.radius - 5) / speed
    target.vy = -40 / time
    assert opportunity(game, sensed(game))[0] == (target,)


def test_opportunity_uses_the_chosen_turn_applied_before_firing(game):
    target = open_scene(game)
    assert opportunity(game, sensed(game), turn=0)[0] == (target,)
    assert not opportunity(game, sensed(game), turn=1)[0]
    game.ship.angle = -game.ship.current_rotation_speed
    assert opportunity(game, sensed(game), turn=1)[0] == (target,)


def test_a_wall_blocking_the_predicted_shot_excludes_an_otherwise_visible_target(game):
    target = open_scene(game)
    target.y += 40
    speed = game.ship.weapon.definition.base_speed
    time = (target.x - game.ship.x - game.ship.radius - 5) / speed
    target.vy = -40 / time
    observations = sensed(game)
    assert opportunity(game, observations)[0] == (target,)
    # The current diagonal sight line is clear; the leading shot along the nose is blocked.
    x = (target.x + game.ship.x) / 2
    game.maze.gate_segments = [WallSegment((x, game.ship.y-8), (x, game.ship.y+8), 3)]
    assert opportunity(game, observations)[0] == ()


def test_cooldown_unseen_targets_empty_ammo_and_invulnerability_are_excluded(game):
    target = open_scene(game)
    observations = sensed(game)
    game.ship.weapon.cooldown = 10
    assert not opportunity(game, observations)[0]
    game.ship.weapon.cooldown = 0
    assert not opportunity(game, np.zeros_like(observations))[0]
    target.indestructible = True
    assert not opportunity(game, observations)[0]
    target.indestructible = False
    game.ship.weapon = None
    from policies.ammo import LimitedAmmo
    game.ship.ammo_policy = LimitedAmmo()
    game.ship.ammo = 0
    assert not opportunity(game, observations)[0]
    game.ship.ammo = 50
    from game_handlers.fire_rate_calculator import calculate_fire_cooldown
    game.last_shot_time = game.clock_ms() - calculate_fire_cooldown(game.ship)
    assert not opportunity(game, observations)[0]  # legacy fire gate uses strictly greater than
    game.fake_now += .001
    assert opportunity(game, observations)[0] == (target,)


def test_weapon_speed_spread_and_lifetime_come_from_the_held_weapon(game):
    target = open_scene(game)
    game.ship.equip_weapon('railgun')
    targets, interval = opportunity(game, sensed(game))
    assert targets == (target,)
    assert interval == pytest.approx(game.ship.weapon.definition.interval / 60)
    game.ship.equip_weapon('grenade_launcher')
    assert not opportunity(game, sensed(game))[0]
    target.y += 40
    game.ship.equip_weapon('basic')
    assert not opportunity(game, sensed(game))[0]
    game.ship.equip_weapon('scattergun')
    assert opportunity(game, sensed(game))[0] == (target,)
    from dataclasses import replace
    from weapons.catalog import WeaponCatalog
    from weapons.state import WeaponInventory
    basic = game.ship.weapon.catalog['basic']
    game.ship.weapon = WeaponInventory(WeaponCatalog([replace(basic, lifetime=1)]))
    target.y = game.ship.y
    assert not opportunity(game, sensed(game))[0]


def test_declining_cost_is_prorated_by_time_not_decision_count_and_firing_is_free():
    from experiments.campaign_rollout import fly_campaign
    from experiments.student import StudentSpec
    from hunter.campaign_pilot import CampaignPilot
    from learning.campaign_rewards import TABLE, CampaignShaping

    class FixedControls:
        def __init__(self, fire):
            self.fire = fire

        def initial_state(self):
            return None

        def act(self, senses, state, *args, **kwargs):
            return (1, False, self.fire), None

    class PracticePilot(CampaignPilot):
        def decide(self, game):
            if self.controller.frames == 0:
                open_scene(game)
                self.interval = game.ship.weapon.definition.interval / 60
            return super().decide(game)

    shaping = CampaignShaping(**{k: 0 for k in TABLE.defaults if k != 'missed_opportunity'},
                              missed_opportunity=-.1)
    episodes = []
    for frames, fire in ((6, False), (12, False), (6, True)):
        pilot = PracticePilot(FixedControls(fire), frames)
        result = fly_campaign(1, 300, StudentSpec('unused'), horizon=.35, pilot=pilot, shaping=shaping)
        expected = 0 if fire else -.1 * .35 / pilot.interval
        assert result['rewards'].sum() == pytest.approx(expected)
        episodes.append(result)
    assert episodes[0]['result'].missed_opportunities == pytest.approx(episodes[1]['result'].missed_opportunities)
    control = fly_campaign(1, 300, StudentSpec('unused'), horizon=.35,
                           pilot=PracticePilot(FixedControls(False), 6), shaping=CampaignShaping())
    np.testing.assert_array_equal(control['senses'], episodes[0]['senses'])
    np.testing.assert_array_equal(control['controls'], episodes[0]['controls'])
    assert control['result'].missed_opportunities is None
