"""The shared reward table: each task's rewards declared once, and everything else derived from it."""
import math
from argparse import ArgumentParser

import numpy as np
import pytest

from learning.reward_table import RewardKind, RewardTable, RewardTerm, tables

TABLE = RewardTable((RewardTerm('kill', RewardKind.REWARD, 10.0, 'Kill', 'points per kill', 'kills'),
                     RewardTerm('crash', RewardKind.PENALTY, -1.0, 'Crash', 'points per crash', 'crashes')),
                    prefix='toy_', help_prefix='Toy: ')


def test_a_table_derives_settings_flags_labels_and_rules():
    assert TABLE.settings == ('toy_kill', 'toy_crash')
    assert TABLE.defaults == {'kill': 10.0, 'crash': -1.0}
    parser = ArgumentParser()
    TABLE.add_arguments(parser)
    args = parser.parse_args(['--toy-crash', '-3'])
    assert TABLE.from_args(args) == {'kill': 10.0, 'crash': -3.0}
    assert TABLE.labels() == {'toy_kill': ('Rewards', 'Kill'), 'toy_crash': ('Rewards', 'Crash')}
    kill, crash = TABLE.reward_rules()
    assert (kill.kind, kill.setting, kill.default) == ('reward', 'toy_kill', 10.0)
    assert (crash.kind, crash.description, crash.default) == ('penalty', 'points per crash', 1.0)
    # A signed table's penalty says on the command line that its points are negative, from its kind alone.
    helps = {a.dest: a.help for a in parser._actions}
    assert helps == {'help': helps['help'], 'toy_kill': 'Toy: points per kill',
                     'toy_crash': 'Toy: points per crash (negative: a penalty)'}


def test_a_term_declares_its_kind_as_a_reward_kind_and_pays_on_an_event_only_if_it_pays():
    with pytest.raises(TypeError):
        RewardTerm('kill', 'reward', 10.0, 'Kill', 'points per kill', 'kills')
    with pytest.raises(ValueError, match='needs the event'):
        RewardTerm('kill', RewardKind.REWARD, 10.0, 'Kill', 'points per kill')
    with pytest.raises(ValueError, match='pays on no event'):
        RewardTerm('range', RewardKind.SHAPING, 200.0, 'Range', 'px', 'ranges')
    # A penalty at 0 is still a penalty: its kind says so, not its value.
    shot = RewardTerm('shot', RewardKind.PENALTY, 0.0, 'Shot', 'points per shot', 'shots')
    assert shot.kind is RewardKind.PENALTY and shot.pays

def test_a_table_pays_each_event_by_name_whatever_order_the_row_holds():
    rows = [{'crashes': 2, 'kills': 0}, {'kills': 1, 'crashes': 0}]
    assert list(TABLE.pay(rows, {'kill': 10.0, 'crash': -1.0})) == [-2.0, 10.0]


def test_a_row_that_does_not_match_the_table_is_an_error():
    with pytest.raises(ValueError, match='missing'):
        TABLE.pay([{'kills': 1}], {})
    with pytest.raises(ValueError, match='unpaid'):
        TABLE.pay([{'kills': 1, 'crashes': 0, 'spins': 3}], {})


def test_a_table_refuses_repeated_keys_or_events_and_bad_values():
    with pytest.raises(ValueError, match='event'):
        RewardTable((RewardTerm('a', RewardKind.REWARD, 1.0, 'A', '', 'x'),
                     RewardTerm('b', RewardKind.REWARD, 1.0, 'B', '', 'x')))
    with pytest.raises(ValueError, match='key'):
        RewardTable((RewardTerm('a', RewardKind.REWARD, 1.0, 'A', '', 'x'),
                     RewardTerm('a', RewardKind.REWARD, 1.0, 'B', '', 'y')))
    with pytest.raises(ValueError, match='finite'):
        TABLE.validate({'kill': math.inf})
    with pytest.raises(ValueError, match='unknown'):
        TABLE.validate({'spin': 1.0})


@pytest.mark.parametrize('key', sorted(tables()))
def test_each_term_pays_its_own_points_on_its_own_event(key):
    """A term's event and weight are aligned: one count of a term's event, and nothing else, pays exactly that term's
    points. Every term gets distinct points, so a weight paid on another term's event shows."""
    table = tables()[key]
    values = {t.key: float(i + 2) * (1 if i % 2 else -1) for i, t in enumerate(table.settable)}
    for term in table.paying:
        row = table.row()
        row[term.event] = 1
        assert table.pay([row], values)[0] == table.weight(term, table.validate(values)), term.key


def test_the_campaign_hunter_is_declared_in_a_table_and_its_rollout_counts_only_its_events():
    from experiments import campaign_rollout
    from learning.campaign_rewards import TABLE as CAMPAIGN
    from learning.tasks import task
    assert task('campaign_hunter').reward_table() is CAMPAIGN
    recorder = campaign_rollout._Recorder.__new__(campaign_rollout._Recorder)
    recorder.events = [CAMPAIGN.row()]
    with pytest.raises(KeyError):
        recorder.add('not_an_event', 1)


def test_the_campaign_rewards_match_the_old_hand_written_sum():
    """The weights the rollout used to line up by position, against the table's sum by name."""
    from learning.campaign_rewards import TABLE as CAMPAIGN, CampaignShaping
    shaping = CampaignShaping(compass_following=0.3, shot=-0.5)
    old_events = ('own_kills', 'pickups', 'beacon', 'wall_crashes', 'following', 'idle', 'exit_progress',
                  'hunt_progress', 'thrusting', 'shots')
    weights = np.array([shaping.own_kill, shaping.pickup, shaping.beacon, shaping.wall_crash,
                        shaping.compass_following, shaping.idle, shaping.exit_progress,
                        shaping.hunt_progress, shaping.thrust, shaping.shot])
    rng = np.random.default_rng(0)
    for end, cleared in (('exited', True), ('exited', False), ('scored_out', False), ('horizon', True)):
        rows = []
        for _ in range(200):
            row = CAMPAIGN.row()
            row.update({e: int(rng.integers(0, 3)) for e in ('own_kills', 'pickups', 'beacon', 'wall_crashes',
                                                              'following', 'idle', 'thrusting', 'shots')})
            row.update(hunt_progress=float(rng.uniform(-0.2, 0.2)), exit_progress=float(rng.uniform(-0.2, 0.2)))
            rows.append(row)
        old = np.asarray([[r[e] for e in old_events] for r in rows], dtype=np.float64) @ weights
        old[-1] += {'exited': shaping.exit + (shaping.cleared_exit if cleared else 0.0),
                    'scored_out': shaping.score_out, 'horizon': 0.0}[end]
        rows[-1].update(exited=int(end == 'exited'), cleared_exit=int(end == 'exited' and cleared),
                        scored_out=int(end == 'scored_out'))
        new = CAMPAIGN.pay(rows, vars(shaping))
        assert np.array_equal(old.astype(np.float32), new.astype(np.float32))


def test_the_campaign_flags_keep_their_names_and_a_run_records_only_its_settings():
    from tools import ppo_neural
    args = ppo_neural.build_parser().parse_args(['--task', 'campaign_hunter', '--shot', '-0.25', '--levels', '1-3'])
    assert {'own_kill', 'cleared_exit', 'score_out', 'compass_following', 'shot'} <= set(vars(args))
    record = ppo_neural.run_record(args)
    assert set(record) == {'task', 'args', 'observation_version', 'decision_frames', 'turn_rate'}
    assert record['args']['shot'] == -0.25
    shaping = ppo_neural.shaping_from(args)
    assert shaping['shot'] == -0.25 and shaping['wall_crash'] == -1.0 and shaping['horizon'] == args.horizon


def test_the_jelly_is_declared_in_a_table_with_its_manifest_fields_and_flags_unchanged():
    from learning.jelly_rewards import TABLE as JELLY, MANIFEST_FIELDS
    from learning.tasks import task
    from tools import ppo_neural
    assert task('jelly_team').reward_table() is JELLY
    assert set(JELLY.settings) <= set(task('jelly_team').options)
    assert MANIFEST_FIELDS['stuck_penalty'] == ('rewards', 'stuck_per_decision', -1)
    assert MANIFEST_FIELDS['catch_reward'] == ('rewards', 'catch', 1)
    assert MANIFEST_FIELDS['no_catch_timeout'] == ('training_limits', 'no_catch_timeout_seconds', 1)
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--jelly-stuck-penalty', '.3',
                                                 '--jelly-max-stage', '1'])
    assert args.jelly_stuck_penalty == .3 and args.jelly_max_stage == 1. and args.jelly_catch_reward == 10.
    # A Jelly's penalties are sizes: a count subtracts them.
    row = JELLY.row()
    row.update(stuck_decisions=1, catches=1)
    assert JELLY.pay([row], {'stuck_penalty': .3})[0] == pytest.approx(10. - .3)
    # Its run settings and shaping parameters are settings, but only the shaping ones are dashboard rules.
    rules = {r.key: r for r in JELLY.reward_rules()}
    assert 'max_stage' not in rules and rules['flank_range'].kind == 'shaping'
    assert rules['flank'].stages == (2, 3) and rules['death'].kind == 'penalty'


def test_the_flocker_is_declared_in_a_table_with_its_flags_manifest_and_checks_unchanged():
    from learning import flocker_rewards
    from learning.tasks import task
    from tools import ppo_neural
    table = flocker_rewards.TABLE
    assert task('flocker_team').reward_table() is table
    # The order a run's manifest records them in, as before.
    assert tuple(flocker_rewards.DEFAULTS) == (
        'hit_reward', 'kill_reward', 'shot_cost', 'death_penalty', 'progress_reward', 'aim_progress_reward',
        'cohesion_reward', 'bunch_penalty', 'stuck_penalty', 'perch_reward', 'volley_reward', 'wall_collision_penalty',
        'bunch_distance', 'contact_only', 'damage_penalty', 'threat_penalty')
    kinds = {t.key: t.kind for t in table.terms}
    assert {k for k, kind in kinds.items() if kind is RewardKind.PENALTY} == {
        'shot_cost', 'death_penalty', 'bunch_penalty', 'stuck_penalty', 'wall_collision_penalty', 'damage_penalty',
        'threat_penalty'}
    assert {k for k, kind in kinds.items() if kind is RewardKind.SHAPING} == {'bunch_distance', 'contact_only'}
    args = ppo_neural.build_parser().parse_args(['--task', 'flocker_team', '--flocker-shot-cost', '.2'])
    assert flocker_rewards.from_args(args)['shot_cost'] == .2
    with pytest.raises(ValueError, match='nonnegative'):
        flocker_rewards.validate({'shot_cost': -1.})
    with pytest.raises(ValueError, match='below one accepted hit'):
        flocker_rewards.validate({'volley_reward': 2.})
    # A Flocker's penalties are sizes: one shot and one hit pay the hit less the shot.
    row = table.row()
    row.update(hits=1, shots=1)
    assert table.pay([row], flocker_rewards.DEFAULTS)[0] == pytest.approx(1. - .05)


def test_the_inchworm_is_declared_in_a_table_with_its_flags_manifest_and_checks_unchanged():
    from learning import inchworm_rewards
    from learning.tasks import task
    from tools import ppo_neural
    table = inchworm_rewards.TABLE
    assert task('inchworm').reward_table() is table
    # The order a run's manifest records them in, as before.
    assert tuple(inchworm_rewards.DEFAULTS) == (
        'progress_reward', 'bite_reward', 'success_reward', 'death_penalty', 'effort_cost', 'airborne_cost',
        'time_cost', 'action_change_cost', 'backward_cost', 'turn_reward', 'dagger_weight', 'speed_bonus',
        'crawl_time_cost', 'about_face_time_cost', 'end_pitch_cost')
    kinds = {t.key: t.kind for t in table.terms}
    assert {k for k, kind in kinds.items() if kind is RewardKind.REWARD} == {
        'progress_reward', 'bite_reward', 'success_reward', 'turn_reward', 'speed_bonus'}
    assert kinds['dagger_weight'] is RewardKind.SHAPING          # a teacher weight, not a reward
    args = ppo_neural.build_parser().parse_args(['--task', 'inchworm', '--inchworm-death-penalty', '7'])
    assert inchworm_rewards.from_args(args)['death_penalty'] == 7.
    with pytest.raises(ValueError, match='nonnegative'):
        inchworm_rewards.validate({'time_cost': -1.})
    with pytest.raises(ValueError, match='unknown'):
        inchworm_rewards.validate({'spin_reward': 1.})
    # An Inchworm's penalties are sizes: dying after one bite pays the bite less the death.
    row = table.row()
    row.update(bites=1, deaths=1)
    assert table.pay([row], inchworm_rewards.DEFAULTS)[0] == pytest.approx(2. - 5.)


def test_the_hunter_is_declared_in_a_table_with_its_fixed_and_scaled_terms():
    from experiments.score import Weights
    from learning.hunter_rewards import TABLE as HUNTER, KILL_REWARD_TOTAL
    from learning.tasks import task
    from tools import ppo_neural
    from tools.trainer import settings
    assert task('hunter').reward_table() is HUNTER
    # The kill reward and the idle cost are fixed: no setting, no flag.
    assert 'kills' not in HUNTER.settings and 'idle' not in HUNTER.settings
    flags = {a.dest for a in ppo_neural.build_parser()._actions}
    assert set(HUNTER.settings) <= flags and 'kills' not in flags
    # The wall and enemy-hit settings scale the score's own weights; the settings form shows them in points.
    values = HUNTER.validate({'wall_scale': 2.0})
    assert HUNTER.points(HUNTER.by_key['wall_scale'], values) == 2.0 * Weights().quotient
    assert settings.BY_DEST['hit_scale'].display_scale == Weights().hit
    # The dashboard shows each term's points as the run's settings pay them: a fixed term's own, a scaled one's in
    # points, a setting's value with its own sign (the page draws the term's direction).
    from learning.adapters import hunter_dashboard
    args = vars(ppo_neural.build_parser().parse_args(['--powerup-reward', '-3', '--death-penalty', '5']))
    shown = hunter_dashboard.recorded({'args': args})
    assert shown['kills'] == KILL_REWARD_TOTAL and shown['wall'] == Weights().quotient
    assert (shown['powerup'], shown['death'], shown['still_speed']) == (-3.0, 5.0, .15)
    assert hunter_dashboard.recorded({'rewards': [{'name': 'Kills', 'value': 1.0}]}) == {}   # no settings: none


def test_a_hunter_episode_pays_each_kill_its_share_of_the_kill_reward():
    import numpy as np
    from experiments import rollout
    from hunter.senses import SENSE_SIZE
    recorder = rollout.RolloutRecorder(6, wall_scale=0.0)
    recorder._start_enemies = 4
    for kills in (1, 0, 2):
        recorder.senses.append(np.zeros(SENSE_SIZE, np.float32))
        recorder.controls.append((1, 0, 0))
        recorder._events.append(rollout.TABLE.row())
        recorder._kills.append(kills)
    assert list(recorder.arrays()[2]) == pytest.approx([25.0, 0.0, 50.0])
