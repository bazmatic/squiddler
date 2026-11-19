"""Campaign episodes: the real ship flies one level headless, and is paid roughly as a good player would be."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.campaign_rollout import CampaignShaping, fly_campaign
from experiments.student import StudentSpec
from hunter.campaign_pilot import ShipCommand
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
from hunter.neural.loader import DEFAULT_PATH as HUNTER_POLICY_PATH
from tools.migrate_policy import migrate

SHAPING = CampaignShaping(idle=0.0, thrust=0.0)       # the terminal rewards read alone; an idle ship would add its penalty


@pytest.fixture(scope='module', autouse=True)
def undo_harness_side_effects():
    yield
    from experiments import harness
    harness.reset()


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / 'campaign.npz'
    migrate(str(HUNTER_POLICY_PATH), str(path), CAMPAIGN_FEATURE_NAMES)
    return StudentSpec(weights=str(path), greedy=False, sample_seed=1)


class ScriptedPilot:
    """Holds still and runs `script(game, decision)` at each decision, reporting blank senses as a pilot would."""

    def __init__(self, script=lambda game, decision: None, thrust=False, turns=lambda decision: 0):
        self.script, self.thrust, self.turns, self.decisions, self._listeners = script, thrust, turns, 0, []

    def on_observed(self, listener):
        self._listeners.append(listener)

    def decide(self, game):
        for listener in self._listeners:
            listener(np.zeros(len(CAMPAIGN_FEATURE_NAMES), dtype=np.float32), (1, 0, 0))
        self.script(game, self.decisions)
        turn = self.turns(self.decisions)
        self.decisions += 1
        return ShipCommand(turn, self.thrust, False)


def into_the_open_exit(game):
    game.exit_opened = True
    game.maze.exit.set_activated(True, game.sound_manager)
    game.ship.x, game.ship.y = game.maze.exit.get_pos()
    game.ship.vx = game.ship.vy = 0.0


def test_a_campaign_episode_is_deterministic_and_well_formed(spec):
    a = fly_campaign(1, 300, spec, horizon=3.0)
    b = fly_campaign(1, 300, spec, horizon=3.0)
    n = len(a['senses'])
    assert n >= 25                                       # 3 s at 10 decisions a second
    assert a['senses'].shape == (n, len(CAMPAIGN_FEATURE_NAMES)) and a['senses'].dtype == np.float32
    assert a['controls'].shape == (n, 3) and a['controls'].dtype == np.int8
    assert a['rewards'].shape == a['temperatures'].shape == (n,)
    assert np.isfinite(a['rewards']).all()
    for key in ('senses', 'controls', 'rewards', 'temperatures'):
        assert np.array_equal(a[key], b[key])
    assert a['result'] == b['result'] and a['result'].end == 'horizon'


def test_exiting_pays_and_clearing_first_pays_more(spec):
    def leave(game, decision):
        if decision == 1:
            into_the_open_exit(game)

    def clear_then_leave(game, decision):
        if decision == 1:
            for enemy in game.roster.alive():
                game.encounter.kill(enemy)
            into_the_open_exit(game)

    left = fly_campaign(1, 300, spec, horizon=5.0, pilot=ScriptedPilot(leave), shaping=SHAPING)
    cleared = fly_campaign(1, 300, spec, horizon=5.0, pilot=ScriptedPilot(clear_then_leave), shaping=SHAPING)
    assert left['result'].end == cleared['result'].end == 'exited'
    assert left['result'].enemies_at_start > 0 and not left['result'].cleared and cleared['result'].cleared
    assert left['rewards'][-1] == pytest.approx(SHAPING.exit)
    assert cleared['rewards'][-1] == pytest.approx(SHAPING.exit + SHAPING.cleared_exit)


def test_idling_costs_a_penalty_a_decision_and_thrusting_does_not(spec):
    idle = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(), shaping=CampaignShaping(exit_progress=0.0, hunt_progress=0.0))
    assert idle['rewards'].sum() == pytest.approx(len(idle['rewards']) * CampaignShaping().idle)
    thrusting = fly_campaign(1, 300, spec, horizon=0.3, pilot=ScriptedPilot(thrust=True),
                         shaping=CampaignShaping(exit_progress=0.0, hunt_progress=0.0, thrust=0.0))
    assert thrusting['rewards'].sum() == pytest.approx(0)


def test_shaking_costs_a_penalty_and_keeping_a_direction_does_not(spec):
    shaping = CampaignShaping(exit_progress=0.0, hunt_progress=0.0, idle=0.0, thrust=0.0, turn_change=-0.5)
    shaking = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(turns=lambda n: 1 if n % 2 else -1), shaping=shaping)
    # the first reversal is free; every one hot on its heels is a shake
    assert shaking['rewards'].sum() == pytest.approx((len(shaking['rewards']) - 2) * shaping.turn_change)
    steady = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(turns=lambda n: 1), shaping=shaping)
    assert steady['rewards'].sum() == pytest.approx(0)


def test_a_lone_reversal_is_free_but_reversing_again_straight_away_costs(spec):
    from learning.campaign_rewards import TABLE
    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'turn_change'}, turn_change=-1.0)

    def fly(turns):
        return fly_campaign(1, 300, spec, horizon=.2, pilot=ScriptedPilot(turns=turns), shaping=shaping)['rewards'].sum()

    assert fly(lambda n: -1 if n < 4 else 0) == pytest.approx(0)                      # stopping a turn is free
    assert fly(lambda n: -1 if n < 4 else 0 if n < 8 else 1) == pytest.approx(0)      # one reversal, after a pause
    assert fly(lambda n: -1 if n < 4 else 1 if n < 8 else -1) == pytest.approx(-1)    # reversed back within a moment
    assert fly(lambda n: -1 if n < 4 else 1 if n < 14 else -1) == pytest.approx(0)    # reversed back after a while


def test_wall_penalty_scales_with_real_collision_severity(spec):
    from learning.campaign_rewards import TABLE
    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'wall_crash'}, wall_crash=-80)

    def collision(normal_share, speed_share=1.0, shielded=False):
        def setup(game, decision):
            if decision != 0:
                return
            maze, ship = game.maze, game.ship
            maze.curved = False
            for y, row in enumerate(maze.grid):
                for x in range(len(row)):
                    row[x] = int(x in (0, 4, maze.grid_width-1) or y in (0, maze.grid_height-1))
            maze.rebuild_walls()
            x, y = maze.position_calculator.grid_center_to_screen(3,5)
            wall_x = maze.position_calculator.grid_to_screen(4,5)[0]
            ship.x, ship.y = wall_x - ship.radius - .01, y
            ship.vx = ship.max_speed * speed_share * normal_share
            ship.vy = ship.max_speed * speed_share * np.sqrt(1-normal_share**2)
            ship.shield_active = False
            ship.shield_initial_timer = 0
            ship.powerup_shield_timer = 60 if shielded else 0
        return fly_campaign(1, 300, spec, horizon=1/60, pilot=ScriptedPilot(setup), shaping=shaping)

    hard, glancing, slow = collision(1), collision(.05), collision(1, .1)
    assert [e['stats']['wall_crashes'] for e in (hard, glancing, slow)] == [1, 1, 1]
    hard_cost, glancing_cost, slow_cost = (-e['rewards'].sum() for e in (hard, glancing, slow))
    assert 0 < glancing_cost < hard_cost * .15
    assert slow_cost == pytest.approx(hard_cost * .1)
    shielded = collision(1, shielded=True)
    assert shielded['rewards'].sum() == 0


def test_the_thruster_costs_a_penalty_a_decision_and_coasting_does_not(spec):
    shaping = CampaignShaping(exit_progress=0.0, hunt_progress=0.0, idle=0.0)
    thrusting = fly_campaign(1, 300, spec, horizon=0.3, pilot=ScriptedPilot(thrust=True), shaping=shaping)
    assert thrusting['rewards'].sum() == pytest.approx(len(thrusting['rewards']) * shaping.thrust)


def test_closing_on_the_open_exit_pays_and_standing_off_does_not(spec):
    def open_then_jump(game, decision):
        if decision == 1:
            for enemy in game.roster.alive():
                game.encounter.kill(enemy)
            game.exit_opened = True
        if decision == 4:                     # a cell from the exit, where the ship started far off
            x, y = game.maze.exit.get_pos()
            game.ship.x, game.ship.y = x + game.maze.cell_size_x, y
            game.ship.vx = game.ship.vy = 0.0

    shaping = CampaignShaping(idle=0.0, thrust=0.0, compass_following=0.0)
    closing = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(open_then_jump), shaping=shaping)
    standing = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(lambda g, d: d == 1 and (list(map(g.encounter.kill, g.roster.alive())), setattr(g, 'exit_opened', True))),
                            shaping=shaping)
    assert closing['result'].end == 'horizon' and closing['rewards'].sum() > 1
    assert standing['rewards'].sum() == pytest.approx(0, abs=1e-6)


def test_closing_on_the_nearest_enemy_pays_and_a_kill_is_not_ground_lost(spec):
    def jump_to_an_enemy(game, decision):
        if decision == 4:
            enemy = game.roster.alive()[0]
            game.ship.x, game.ship.y = enemy.x + 90, enemy.y
            game.ship.vx = game.ship.vy = 0.0

    def kill_one(game, decision):
        if decision == 4:
            game.encounter.kill(game.roster.alive()[0])

    shaping = CampaignShaping(idle=0.0, thrust=0.0, compass_following=0.0, own_kill=0.0)
    closing = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(jump_to_an_enemy), shaping=shaping)
    killing = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(kill_one), shaping=shaping)
    assert closing['rewards'].sum() > 0
    assert killing['rewards'].min() >= 0


def test_approaching_enemies_do_not_pay_hunt_progress_to_a_stationary_ship(spec):
    from learning.campaign_rewards import TABLE
    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'hunt_progress'}, hunt_progress=40.0)
    positions = []
    target = []

    def approaching(game, decision):
        if decision == 0:
            enemy = max(game.roster.alive(), key=lambda e: np.hypot(e.x-game.ship.x, e.y-game.ship.y))
            target.append(enemy)
            for other in game.roster.alive():
                if other is not enemy:
                    game.encounter.kill(other)
        if decision == 4:
            target[0].x, target[0].y = game.ship.x + 100, game.ship.y

    episode = fly_campaign(1, 300, spec, horizon=.2, pilot=ScriptedPilot(approaching), shaping=shaping,
                           trace=lambda seconds, game: positions.append(game.ship.get_pos()))
    assert len(set(positions)) == 1
    assert episode['rewards'].sum() == pytest.approx(0, abs=1e-6)


def test_only_the_ships_own_kills_pay(spec):
    def killed_by(source):
        def script(game, decision):
            if decision == 1:
                game.encounter.kill(game.roster.alive()[0], SimpleNamespace(source=source, weapon_id=None))
        return script

    ally = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(killed_by('hunter')))
    own = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(killed_by('player')))
    assert (ally['result'].own_kills, ally['result'].ally_kills) == (0, 1)
    assert (own['result'].own_kills, own['result'].ally_kills) == (1, 0)
    assert own['rewards'].sum() - ally['rewards'].sum() == pytest.approx(SHAPING.own_kill)


def test_unseen_pursuit_pays_for_ship_progress_but_not_enemy_motion_or_visible_combat(spec):
    from learning.campaign_rewards import TABLE

    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'unseen_progress'},
                              unseen_progress=20.0)

    def approach(game, decision):
        if decision == 4:
            enemy = game.roster.alive()[0]
            game.ship.x, game.ship.y = enemy.x + 90, enemy.y
            game.ship.vx = game.ship.vy = 0.0

    class VisiblePilot(ScriptedPilot):
        def on_observed(self, listener):
            def observed(senses, controls):
                senses[CAMPAIGN_FEATURE_NAMES.index('enemy0_present')] = 1
                listener(senses, controls)
            super().on_observed(observed)

    hidden = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(approach), shaping=shaping)
    visible = fly_campaign(1, 300, spec, horizon=1.0, pilot=VisiblePilot(approach), shaping=shaping)
    stationary = fly_campaign(1, 300, spec, horizon=1.0, pilot=ScriptedPilot(), shaping=shaping)
    assert hidden['rewards'].sum() > 0
    assert visible['rewards'].sum() == pytest.approx(0, abs=1e-6)
    assert stationary['rewards'].sum() == pytest.approx(0, abs=1e-6)


def test_search_lessons_start_without_sight_and_record_contact_time(spec):
    hidden = fly_campaign(10, 300, spec, horizon=.2)
    visible = fly_campaign(1, 300, spec, horizon=.2)
    assert hidden['result'].started_unseen
    assert hidden['result'].first_sighting_seconds is None
    assert hidden['result'].unseen_seconds > 0
    assert not visible['result'].started_unseen
    assert visible['result'].first_sighting_seconds == pytest.approx(0)


def test_a_pursuit_lesson_pays_contact_once_and_only_after_an_unseen_start(spec):
    from hunter.campaign_pilot import CampaignPilot
    from hunter.neural.policy import NeuralPolicy
    from learning.campaign_rewards import TABLE
    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'search_contact'}, search_contact=100)

    class FindEnemy(CampaignPilot):
        move_at = 6

        def decide(self, game):
            if self.controller.frames == self.move_at:
                enemy = game.roster.alive()[0]
                game.ship.x, game.ship.y = enemy.x + 30, enemy.y
                game.ship.vx = game.ship.vy = 0
            return super().decide(game)

    pilot = FindEnemy(NeuralPolicy.load(spec.weights, CAMPAIGN_FEATURE_NAMES), 6)
    contact = fly_campaign(10, 300, spec, horizon=1, pursuit_lesson=True, shaping=shaping, pilot=pilot)
    assert contact['result'].started_unseen and contact['result'].end == 'contact'
    assert contact['result'].elapsed < 1
    assert contact['rewards'].sum() == 100
    class AlreadyThere(FindEnemy):
        move_at = 0

    visible = fly_campaign(10, 300, spec, horizon=1, pursuit_lesson=True, shaping=shaping,
                           pilot=AlreadyThere(NeuralPolicy.load(spec.weights, CAMPAIGN_FEATURE_NAMES), 6))
    assert visible['result'].end == 'visible_start'
    assert visible['reward_totals']['search_contact'] == 0
    normal = fly_campaign(1, 300, spec, horizon=.2, shaping=shaping)
    assert normal['result'].end != 'visible_start'
    assert normal['reward_totals']['search_contact'] == 0


def test_recorded_reward_components_add_to_the_actual_episode_return(spec):
    episode = fly_campaign(1, 300, spec, horizon=1)
    assert sum(episode['reward_totals'].values()) == pytest.approx(episode['rewards'].sum(), abs=1e-4)


def test_waiting_cannot_complete_a_pursuit_lesson_and_the_target_has_an_open_route(spec):
    from hunter.campaign_pilot import CampaignPilot
    from hunter.maze_route import MazeRoutes

    class Still:
        def initial_state(self):
            return None

        def act(self, senses, state, *args, **kwargs):
            return (1, False, False), None

    positions = []
    def observe(seconds, game):
        target = game.roster.alive()[0]
        positions.append(target.get_pos())
        if len(positions) == 1:
            assert MazeRoutes().route(game.maze, game.ship.get_pos(), [target.get_pos()]) is not None
            assert not game.powerup_crystals

    episode = fly_campaign(10, 400, spec, horizon=3, pursuit_lesson=True,
                           pilot=CampaignPilot(Still(), 6), trace=observe)
    assert len(set(positions)) == 1
    assert episode['result'].started_unseen and episode['result'].first_sighting_seconds is None
    assert episode['result'].end == 'horizon'


def test_pursuit_placement_rejects_targets_hidden_only_by_moving_gates(spec):
    with pytest.raises(ValueError, match='no hidden pursuit target'):
        fly_campaign(22, 400, spec, horizon=1/60, pursuit_lesson=True)


def test_retreating_during_an_unseen_search_costs_progress(spec):
    from learning.campaign_rewards import TABLE

    class RetreatPilot(ScriptedPilot):
        def decide(self, game):
            if self.decisions == 0:
                self.start = game.ship.get_pos()
                enemy, *others = game.roster.alive()
                for other in others:
                    game.encounter.kill(other)
                game.ship.x, game.ship.y = enemy.x + 90, enemy.y
            if self.decisions == 4:
                game.ship.x, game.ship.y = self.start
            return super().decide(game)

    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'unseen_progress'},
                              unseen_progress=20.0)
    episode = fly_campaign(1, 300, spec, horizon=.2, pilot=RetreatPilot(), shaping=shaping)
    assert episode['rewards'].sum() < 0


@pytest.mark.parametrize('bearing,penalized', [(None, True), (15, False), (40, True)])
def test_campaign_reuses_the_hunters_visible_20_degree_shot_check(spec, bearing, penalized):
    from learning.campaign_rewards import TABLE

    class FiringPilot(ScriptedPilot):
        def on_observed(self, listener):
            def observed(senses, controls):
                if bearing is not None:
                    for part, value in (('present', 1), ('sin', np.sin(np.radians(bearing))),
                                        ('cos', np.cos(np.radians(bearing)))):
                        senses[CAMPAIGN_FEATURE_NAMES.index(f'enemy0_{part}')] = value
                listener(senses, controls)
            super().on_observed(observed)

        def decide(self, game):
            command = super().decide(game)
            return ShipCommand(command.turn, command.thrust, True)

    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'unaimed_shot'}, unaimed_shot=-.75)
    flown = fly_campaign(1, 300, spec, horizon=.6, pilot=FiringPilot(), shaping=shaping)
    shots = flown['stats'].get('shots', 0)
    assert shots > 0
    assert flown['result'].unaimed_shots == (shots if penalized else 0)
    assert flown['rewards'].sum() == pytest.approx(-.75 * shots if penalized else 0)


@pytest.mark.parametrize('source,damage,points', [('player', 1, 2), ('hunter', 1, 0), ('player', 0, 0)])
def test_campaign_credits_only_the_players_damaging_hits(spec, source, damage, points):
    from learning.campaign_rewards import TABLE

    def hit(game, decision):
        if decision == 4:
            game.encounter.apply_hit(game.roster.alive()[0], SimpleNamespace(
                source=source, weapon_id=None, vx=0, vy=0, creature_damage=damage))

    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'enemy_hit'}, enemy_hit=2.0)
    flown = fly_campaign(1, 300, spec, horizon=.2, pilot=ScriptedPilot(hit), shaping=shaping)
    assert flown['rewards'].sum() == pytest.approx(points)
    assert flown['stats'].get('damaging_hits', 0) == (1 if points else 0)


def test_the_longest_spell_without_a_kill_counts_only_while_enemies_remain(spec):
    def ally_kills_one(game, decision):
        if decision == 60:                                  # a second in (one call a frame)
            game.encounter.kill(game.roster.alive()[0], SimpleNamespace(source='hunter', weapon_id=None))

    def clears_the_level(game, decision):
        if decision == 30:                                  # half a second in
            for enemy in game.roster.alive():
                game.encounter.kill(enemy, SimpleNamespace(source='player', weapon_id=None))

    still = fly_campaign(1, 300, spec, horizon=2.0, pilot=ScriptedPilot())['result']
    late = fly_campaign(1, 300, spec, horizon=2.5, pilot=ScriptedPilot(ally_kills_one))['result']
    cleared = fly_campaign(1, 300, spec, horizon=2.0, pilot=ScriptedPilot(clears_the_level))['result']
    assert still.enemies_at_start > 1 and still.own_kills == still.ally_kills == 0
    assert still.max_no_kill_gap == pytest.approx(2.0, abs=0.05)
    assert late.ally_kills == 1 and late.max_no_kill_gap == pytest.approx(1.5, abs=0.15)
    assert cleared.cleared and cleared.max_no_kill_gap == pytest.approx(0.5, abs=0.15)


def test_scoring_out_ends_the_episode_with_its_penalty(spec, monkeypatch):
    from scoring.system import ScoringSystem
    monkeypatch.setattr(ScoringSystem, 'calculate_current_potential_score',
                        lambda self, *a, **k: {'potential_score': 0, 'score_percentage': 0.0})
    r = fly_campaign(1, 300, spec, horizon=5.0, pilot=ScriptedPilot(), shaping=SHAPING)
    assert r['result'].end == 'scored_out' and r['result'].elapsed < 1.0
    assert r['rewards'][-1] == pytest.approx(SHAPING.score_out)


def test_a_fetched_beacon_summons_the_neural_ally_whatever_the_setting(spec, monkeypatch):
    # Jev's pilot reads wall time and calls out: an episode must not depend on it.
    from config import SETTINGS
    from entities.jev_beacon import JevBeacon
    monkeypatch.setattr(SETTINGS.game, 'hunterPilot', 'jev')
    allies = []

    def fetch_beacon(game, decision):
        if decision == 1:
            game.powerup_crystals.append(JevBeacon((game.ship.x, game.ship.y)))
        if decision == 120 and game.hunter_flight is not None:      # a beacon forms for 72 frames first
            allies.append(getattr(game.hunter_flight.controller, 'kind', 'worker'))

    r = fly_campaign(13, 300, spec, horizon=2.5, pilot=ScriptedPilot(fetch_beacon))
    assert r['result'].beacon and allies == ['neural']
    assert SETTINGS.game.hunterPilot == 'jev'


def test_one_campaign_training_iteration_saves_a_campaign_policy(tmp_path):
    pytest.importorskip('torch')
    from hunter.neural.policy import NeuralPolicy
    from tools import ppo_neural
    lines = []
    result = ppo_neural.train_rl(str(HUNTER_POLICY_PATH), tmp_path, iterations=1, episodes=2, workers=1,
                                 task='campaign_hunter', levels=[1], horizon=2, log=lines.append)
    NeuralPolicy.load(result['last'], CAMPAIGN_FEATURE_NAMES)        # raises InvalidPolicy if it is not one
    assert result['last'].name == 'iter0000.npz'
    for key in ('exited=', 'cleared_exit=', 'scored_out=', 'own_kills=', 'mean_return='):
        assert key in lines[0]


def test_the_campaign_rewards_are_trainer_settings_for_that_task_alone():
    from dataclasses import fields
    from tools import ppo_neural
    from tools.trainer.settings import applies_to
    args = ppo_neural.build_parser().parse_args(['--task', 'campaign_hunter'])
    for field in fields(CampaignShaping):
        assert getattr(args, field.name) == field.default
        assert applies_to(field.name) == ('campaign_hunter',)


def test_the_trainer_server_still_starts_without_pygame():
    import subprocess
    import sys
    loaded = subprocess.run([sys.executable, '-c', "import sys, tools.trainer.server; print('pygame' in sys.modules)"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert loaded == 'False'


def test_exit_with_crawler_practice_uses_level_32_and_preserves_campaign_layout(spec):
    import math
    import level_config
    from learning.campaign_training_levels import EXIT_WITH_CRAWLER
    from hunter.maze_route import MazeRoutes
    original = level_config.get_level_layout(32)
    seen = []

    def inspect(seconds, game):
        if seen:
            return
        alive = game.roster.alive()
        origin = game.ship.get_pos()
        enemy = alive[0]
        route = MazeRoutes().route(game.maze, origin, [enemy.get_pos()])
        seen.append((game.level, game.exit_opened, len(alive), enemy.kind.key, origin,
                     game.maze.exit.get_pos(), enemy.get_pos(), route))

    episode = fly_campaign(EXIT_WITH_CRAWLER, 399, spec, horizon=.2, trace=inspect, sticky_targets=True,
                           target_recovery=True)
    level, exit_open, count, kind, origin, exit_pos, enemy_pos, route = seen[0]
    assert episode['result'].level == EXIT_WITH_CRAWLER
    assert level == 32 and exit_open and count == 1 and kind == 'wallcrawler'
    senses = episode['senses'][0]
    assert senses[CAMPAIGN_FEATURE_NAMES.index('enemy0_present')] == 0
    assert senses[CAMPAIGN_FEATURE_NAMES.index('enemy_compass_present')] == 1
    assert senses[CAMPAIGN_FEATURE_NAMES.index('exit_open')] == 1
    assert math.dist(origin, exit_pos) < 400
    assert math.dist(origin, enemy_pos) > 300
    assert route is not None
    assert level_config.get_level_layout(32) == original
    assert len(original.enemies) == 20


def test_exit_with_crawler_practice_exits_without_killing_crawler(spec):
    from learning.campaign_training_levels import EXIT_WITH_CRAWLER
    episode = fly_campaign(EXIT_WITH_CRAWLER, 399, spec, horizon=.2,
                           pilot=ScriptedPilot(lambda game, decision: into_the_open_exit(game)))
    assert episode['result'].end == 'exited'
    assert episode['result'].enemies_at_start == 1
    assert not episode['result'].cleared
