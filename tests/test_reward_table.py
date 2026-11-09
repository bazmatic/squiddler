"""The shared reward table: each task's rewards declared once, and everything else derived from it."""
import math
from argparse import ArgumentParser

import numpy as np
import pytest

from learning.reward_table import RewardTable, RewardTerm, tables

TABLE = RewardTable((RewardTerm('kill', 10.0, 'Kill', 'points per kill', 'kills'),
                     RewardTerm('crash', -1.0, 'Crash', 'points per crash (negative: a penalty)', 'crashes')),
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
        RewardTable((RewardTerm('a', 1.0, 'A', '', 'x'), RewardTerm('b', 1.0, 'B', '', 'x')))
    with pytest.raises(ValueError, match='key'):
        RewardTable((RewardTerm('a', 1.0, 'A', '', 'x'), RewardTerm('a', 1.0, 'B', '', 'y')))
    with pytest.raises(ValueError, match='finite'):
        TABLE.validate({'kill': math.inf})
    with pytest.raises(ValueError, match='unknown'):
        TABLE.validate({'spin': 1.0})


@pytest.mark.parametrize('key', sorted(tables()))
def test_each_term_pays_its_own_points_on_its_own_event(key):
    """A term's event and weight are aligned: one count of a term's event, and nothing else, pays exactly that term's
    points. Every term gets distinct points, so a weight paid on another term's event shows."""
    table = tables()[key]
    values = {t.key: float(i + 2) * (1 if i % 2 else -1) for i, t in enumerate(table.terms)}
    for term in table.terms:
        row = table.row()
        row[term.event] = 1
        assert table.pay([row], values)[0] == values[term.key], term.key


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


def test_the_campaign_flags_and_recorded_rows_keep_their_names():
    from tools import ppo_neural
    args = ppo_neural.build_parser().parse_args(['--task', 'campaign_hunter', '--shot', '-0.25'])
    assert {'own_kill', 'cleared_exit', 'score_out', 'compass_following', 'shot'} <= set(vars(args))
    rows = {row['name']: row for row in ppo_neural.table_reward_summary(args)}
    assert rows['Own kill'] == {'name': 'Own kill', 'sign': '+', 'value': 10.0,
                                'unit': 'points per enemy the ship destroys itself'}
    assert rows['Shot']['sign'] == '-' and rows['Shot']['value'] == 0.25
    assert 'Compass following' not in rows                       # 0: not recorded
    shaping = ppo_neural.shaping_from(args)
    assert shaping['shot'] == -0.25 and shaping['wall_crash'] == -1.0 and shaping['horizon'] == args.horizon
