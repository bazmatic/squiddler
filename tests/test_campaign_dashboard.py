"""The Campaign Hunter's dashboard: what the trainer page charts from its training log, and its rewards panel."""
from types import SimpleNamespace

from learning import dashboard
from learning.adapters import campaign_hunter

# What tools/ppo_neural.py writes to every campaign train.log row besides the adapter's summary().
PPO_KEYS = {'policy_loss', 'value_loss', 'kl', 'entropy', 'clip_fraction', 'iteration', 'mean_return',
            'rollout_s', 'update_s'}


def episode(end, cleared=False, own_kills=0):
    return dict(result=SimpleNamespace(end=end, cleared=cleared, own_kills=own_kills))


def test_a_campaign_log_row_charts_exits_clears_score_outs_kills_and_the_shared_charts():
    row = campaign_hunter.summary([episode('exited', True, 6), episode('exited', False, 3), episode('scored_out')])
    keys = set(row) | PPO_KEYS
    charted = dashboard.charted('campaign_hunter', keys)
    assert [m['key'] for m in charted] == ['mean_return', 'exited', 'cleared_exit', 'scored_out', 'own_kills',
                                           'kl', 'entropy']
    assert dashboard.unclaimed('campaign_hunter', keys) == []   # nothing left over for "Other"


def test_a_row_missing_a_metric_charts_only_what_it_has():
    charted = dashboard.charted('campaign_hunter', {'mean_return', 'exited', 'own_kills', 'kl', 'entropy'})
    assert [m['key'] for m in charted] == ['mean_return', 'exited', 'own_kills', 'kl', 'entropy']


def test_the_rewards_panel_shows_the_runs_campaign_points_with_score_out_as_a_penalty():
    config = dict(task='campaign_hunter', args=dict(own_kill=10.0, exit=100.0, cleared_exit=100.0, score_out=-80.0,
                                                    wall_crash=-1.0, pickup=0.0),
                  episode=dict(horizon_seconds=180.0, episodes_per_iteration=96, levels=[1, 2, 3]))
    rules = {r['key']: r for r in dashboard.rewards('campaign_hunter', config, {})['rules']}
    assert (rules['score_out']['kind'], rules['score_out']['value']) == ('penalty', 80.0)   # the page draws −80
    assert (rules['own_kill']['kind'], rules['own_kill']['value']) == ('reward', 10.0)
    assert rules['pickup']['active'] is False
    assert rules['beacon']['recorded'] is False and rules['beacon']['value'] == 5.0   # not recorded: the default
