"""The Campaign Hunter's dashboard: what the trainer page charts from its training log, and its rewards panel."""
from types import SimpleNamespace

import numpy as np

from learning import dashboard
from learning.adapters import campaign_hunter

# What tools/ppo_neural.py writes to every campaign train.log row besides the adapter's summary().
PPO_KEYS = {'policy_loss', 'value_loss', 'kl', 'entropy', 'clip_fraction', 'iteration', 'mean_return',
            'rollout_s', 'update_s'}


def episode(end, cleared=False, own_kills=0, thrusts=(1, 1), wall_crashes=0, shots=0):
    controls = np.array([(1, t, 0) for t in thrusts], dtype=np.int8)
    return dict(result=SimpleNamespace(end=end, cleared=cleared, own_kills=own_kills, wall_crashes=wall_crashes, shots=shots), controls=controls)


def test_a_campaign_log_row_charts_exits_clears_score_outs_kills_and_the_shared_charts():
    row = campaign_hunter.summary([episode('exited', True, 6), episode('exited', False, 3), episode('scored_out')])
    keys = set(row) | PPO_KEYS
    charted = dashboard.charted('campaign_hunter', keys)
    assert [m['key'] for m in charted] == ['mean_return', 'exited', 'cleared_exit', 'scored_out', 'own_kills',
                                           'shots', 'thrust', 'spinning', 'kl', 'entropy']
    assert dashboard.unclaimed('campaign_hunter', keys) == []   # nothing left over for "Other"


def test_the_thrust_chart_is_the_share_of_decisions_with_the_thruster_on():
    row = campaign_hunter.summary([episode('exited', thrusts=(1, 1, 0, 0)), episode('exited', thrusts=(1, 1, 1, 0))])
    assert row['thrust'] == 0.625


def test_the_wall_strikes_chart_is_the_average_strikes_per_episode():
    row = campaign_hunter.summary([episode('exited', wall_crashes=2), episode('scored_out', wall_crashes=5)])
    assert row['wall_crashes'] == 3.5


def test_impact_severity_is_reported_separately_from_contact_count():
    from tools.campaign_eval import summarize
    first, second = episode('exited', wall_crashes=2), episode('scored_out', wall_crashes=2)
    first['result'].wall_impulse = .2
    second['result'].wall_impulse = 1.8
    row = campaign_hunter.summary([first, second])
    assert row['wall_crashes'] == 2
    assert row['wall_impulse'] == 1
    assert dashboard.unclaimed('campaign_hunter', set(row)) == []
    rows = [dict(vars(e['result']), elapsed=10, enemies_at_start=4, max_no_kill_gap=5)
            for e in (first, second)]
    assert summarize(rows)['wall_impulse'] == 1


def test_wall_chart_shows_points_lost_for_impact_severity_not_contact_count():
    config = dict(task='campaign_hunter', args=dict(wall_crash=-80))
    rows = [dict(iteration=0, wall_crashes=5, wall_impulse=.25),
            dict(iteration=1, wall_crashes=5, wall_impulse=2)]
    charted = dashboard.chart_rows('campaign_hunter', config, rows)
    assert [r['wall_penalty'] for r in charted] == [20, 160]
    assert 'wall_penalty' not in rows[0]  # cached log rows are shared by the server
    metrics = dashboard.charted('campaign_hunter', set(charted[0]))
    assert [m['key'] for m in metrics] == ['wall_penalty']
    assert dashboard.unclaimed('campaign_hunter', set(charted[0])) == []


def test_wall_chart_uses_historical_coefficients_and_old_flat_collision_costs():
    config = dict(args=dict(wall_crash=-20), resume_settings_changes=[
        dict(after_iteration=1, changes=[dict(setting='wall_crash', previous=-80, value=-40)]),
        dict(after_iteration=3, changes=[dict(setting='wall_crash', previous=-40, value=-20)])])
    rows = [dict(iteration=i, wall_impulse=.5) for i in range(5)]
    derived = dashboard.chart_rows('campaign_hunter', config, rows)
    assert [r['wall_penalty'] for r in derived] == [40, 40, 20, 20, 10]
    legacy = dashboard.chart_rows('campaign_hunter', dict(args=dict(wall_crash=-80)),
                                  [dict(iteration=0, wall_crashes=3)])
    assert legacy[0]['wall_penalty'] == 240
    assert dashboard.chart_rows('campaign_hunter', dict(args=dict(wall_crash=0)), rows)[0]['wall_penalty'] == 0


def test_wall_chart_does_not_invent_a_penalty_without_recorded_weights_or_impacts():
    rows = [dict(iteration=0, wall_impulse=2)]
    assert 'wall_penalty' not in dashboard.chart_rows('campaign_hunter', {}, rows)[0]
    assert 'wall_penalty' not in dashboard.chart_rows('campaign_hunter', dict(args=dict(wall_crash=-80)),
                                                     [dict(iteration=0)])[0]


def test_the_bullets_chart_is_the_average_bullets_fired_per_episode():
    row = campaign_hunter.summary([episode('exited', shots=10), episode('scored_out', shots=30)])
    assert row['shots'] == 20.0


def test_search_charts_count_missed_contacts_as_the_whole_episode():
    found, missed = episode('exited'), episode('horizon')
    found['result'].__dict__.update(started_unseen=True, first_sighting_seconds=2.0, elapsed=10, unseen_seconds=3)
    missed['result'].__dict__.update(started_unseen=True, first_sighting_seconds=None, elapsed=10, unseen_seconds=10,
                                   search_horizon_seconds=60)
    row = campaign_hunter.summary([found, missed])
    assert row['search_found'] == .5 and row['search_seconds'] == 31.0
    assert row['unseen_seconds'] == 6.5
    assert dashboard.unclaimed('campaign_hunter', set(row)) == []


def test_progress_charts_report_actual_signed_reward_points_not_settings():
    first, second = episode('horizon'), episode('horizon')
    first['reward_totals'] = dict(hunt_progress=10, unseen_progress=5, exit_progress=0)
    second['reward_totals'] = dict(hunt_progress=-2, unseen_progress=-1, exit_progress=4)
    row = campaign_hunter.summary([first, second])
    assert row['hunt_progress_reward'] == 4
    assert row['unseen_progress_reward'] == 2
    assert row['exit_progress_reward'] == 2
    assert dashboard.unclaimed('campaign_hunter', set(row)) == []
    assert 'hunt_progress_reward' not in campaign_hunter.summary([episode('horizon')])


def test_declined_shot_chart_tracks_elapsed_opportunities_and_actual_penalty():
    first, second = episode('horizon'), episode('horizon')
    for item, count in ((first, .5), (second, 1.5)):
        item['result'].missed_opportunities = count
        item['reward_totals'] = dict(hunt_progress=0, unseen_progress=0, exit_progress=0,
                                     missed_opportunity=-.1*count)
    row = campaign_hunter.summary([first, second])
    assert row['missed_opportunities'] == 1
    assert row['missed_opportunity_reward'] == -.1
    assert dashboard.unclaimed('campaign_hunter', set(row)) == []


def test_accuracy_chart_distinguishes_disabled_rules_from_a_real_zero_reward():
    rows = [dict(iteration=0, shot_accuracy_reward=0)]
    legacy = dict(task='campaign_hunter', args={}, rules={'graded_shooting': False})
    enabled = dict(task='campaign_hunter', args={}, rules={'graded_shooting': True})
    assert 'shot_accuracy_reward' not in dashboard.chart_rows('campaign_hunter', legacy, rows)[0]
    assert dashboard.chart_rows('campaign_hunter', enabled, rows)[0]['shot_accuracy_reward'] == 0
    assert rows[0]['shot_accuracy_reward'] == 0
    item = episode('horizon')
    item['reward_totals'] = dict(hunt_progress=0, unseen_progress=0, exit_progress=0,
                                 missed_opportunity=0, shot_accuracy=0)
    item['result'].graded_shooting = False
    assert 'shot_accuracy_reward' not in campaign_hunter.summary([item])
    item['result'].graded_shooting = True
    assert campaign_hunter.summary([item])['shot_accuracy_reward'] == 0


def test_mixed_practice_keeps_combat_outcomes_and_lesson_contact_rates_separate():
    lesson, combat = episode('contact'), episode('exited', cleared=True, own_kills=8)
    lesson['result'].pursuit_lesson = True
    combat['result'].pursuit_lesson = False
    row = campaign_hunter.summary([lesson, combat])
    assert row['lesson_contact'] == 1
    assert row['exited'] == row['cleared_exit'] == 1
    assert row['own_kills'] == 8


def test_shooting_charts_count_damage_and_weight_unaimed_share_by_bullets():
    first, second = episode('exited', shots=5), episode('horizon', shots=15)
    first['result'].__dict__.update(damaging_hits=2, unaimed_shots=4)
    second['result'].__dict__.update(damaging_hits=3, unaimed_shots=6)
    row = campaign_hunter.summary([first, second])
    assert row['damaging_hits'] == 2.5
    assert row['unaimed_shot_share'] == .5
    assert dashboard.unclaimed('campaign_hunter', set(row)) == []
    first['result'].shots = first['result'].unaimed_shots = 0
    assert campaign_hunter.summary([first])['unaimed_shot_share'] == 0


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


def test_spinning_counts_only_long_unbroken_turns_one_way():
    from learning.adapters.campaign_hunter import SPIN_DECISIONS, spinning_share
    n = SPIN_DECISIONS
    assert spinning_share([2] * n + [1] * n) == 0.5                 # a long turn is a spin
    assert spinning_share(([2] * (n - 1) + [1]) * 3) == 0                         # steering in bursts is not
    assert spinning_share([0, 2] * n) == 0                                        # shaking is not a spin either
    assert spinning_share([]) == 0
