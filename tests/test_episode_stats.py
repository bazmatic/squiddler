"""Any named stat an episode records is averaged into the iteration's log row and charted, with no other change."""
import json
import shutil

import pytest

from learning import dashboard
from learning.adapters import campaign_hunter
from learning.episode_stats import collecting, record, recorded_means
from tests.test_campaign_dashboard import episode as campaign_episode
from tests.test_campaign_training import spec, undo_harness_side_effects  # noqa: F401


def test_a_recorded_stat_is_the_mean_per_episode_and_missing_counts_as_zero():
    with collecting() as first:
        record('Wall strikes')
        record('wall strikes', 2)
    with collecting() as second:
        record('bullets fired', 4)
    assert first == {'wall_strikes': 3}
    assert recorded_means([first, second, None]) == {'wall_strikes': 1.0, 'bullets_fired': 4 / 3}


def test_recording_outside_an_episode_does_nothing():
    record('wall strikes')   # no error, and nothing leaks into the next episode
    with collecting() as stats:
        pass
    assert stats == {}


def test_a_campaign_stat_reaches_the_log_row_and_a_new_key_is_charted_under_other():
    flown = [dict(campaign_episode('exited'), stats={'wall_strikes': 3}), dict(campaign_episode('exited'), stats={})]
    row = campaign_hunter.summary(flown)
    assert row['wall_strikes'] == 1.5
    assert dashboard.unclaimed('campaign_hunter', set(row)) == ['wall_strikes']


def test_recorded_stats_never_displace_the_campaign_hunters_own_numbers():
    plain = campaign_hunter.summary([campaign_episode('exited', wall_crashes=2)])
    clashing = campaign_hunter.summary([dict(campaign_episode('exited', wall_crashes=2), stats={'wall_crashes': 9})])
    assert clashing == plain
    assert list(campaign_hunter.summary([campaign_episode('exited')])) == list(campaign_hunter.summary(
        [dict(campaign_episode('exited'), stats={})]))   # no recorded stat: the charts and their order are as before


def test_a_declared_metric_overrides_the_default_title_section_and_digits():
    assert dashboard.unclaimed('campaign_hunter', {'shots'}) == []   # declared, so charted under its own section
    assert [m['key'] for m in dashboard.charted('campaign_hunter', {'shots'})] == ['shots']


def test_a_stat_recorded_during_a_flown_campaign_episode_is_in_the_episode(spec):
    from experiments.campaign_rollout import CampaignShaping, fly_campaign
    from learning.campaign_rewards import TABLE
    from tests.test_campaign_training import ScriptedPilot
    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults})
    episode = fly_campaign(1, 300, spec, horizon=.2, shaping=shaping,
                           pilot=ScriptedPilot(lambda game, decision: record('decisions made')))
    assert episode['stats']['decisions_made'] > 0
    assert campaign_hunter.summary([episode])['decisions_made'] == episode['stats']['decisions_made']


def test_another_tasks_rollout_carries_recorded_stats_into_its_episode_row(tmp_path, monkeypatch):
    from learning.adapters import jelly_scenario as scenario
    from learning.policy import random_policy
    from learning.run import rollout_job
    from tools import ppo_neural
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--horizon', '.4'])
    shutil.copyfile(args.opponent, tmp_path / 'opponent.npz')
    args.opponent = str(tmp_path / 'opponent.npz')
    manifest = scenario.metadata(args)
    manifest['training'] = {}
    (tmp_path / 'config.json').write_text(json.dumps(dict(task=args.task, args=vars(args), manifest=manifest)))
    random_policy(scenario.SPEC, 3, hidden=8, embed=4).save(tmp_path / 'initial.npz')
    step = scenario.JellyEnvironment.step

    def recording_step(self, *a, **k):
        record('jelly steps')
        return step(self, *a, **k)
    monkeypatch.setattr(scenario.JellyEnvironment, 'step', recording_step)
    case = dict(level=11, seed=301, team_size=1, stage=1)
    _, events, _ = rollout_job(dict(task=args.task, weights=str(tmp_path / 'initial.npz'), opponent=args.opponent,
                                    horizon=args.horizon, muted=False, mode='greedy', case=case))
    assert events['stats']['jelly_steps'] > 0
    rows = [dict(events, **case)]
    assert recorded_means(r.get('stats') for r in rows)['jelly_steps'] == events['stats']['jelly_steps']
    json.dumps(rows)   # the episode record is written as JSON
