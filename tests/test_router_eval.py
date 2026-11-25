"""The held-out evaluation of the router."""
from learning.adapters import inchworm as adapter
from learning.policy import random_policy
from tools.router_eval import LESSONS, evaluate, handover_rate, trial_level, trial_seeds
from learning.router import RoutedPolicy


def test_evaluate_reports_a_success_rate_between_zero_and_one():
    policy = random_policy(adapter.SPEC,1,hidden=8,embed=4)
    assert 0. <= evaluate(policy,'crawl',2,'train') <= 1.


def test_the_two_seed_sets_do_not_overlap():
    for lesson in LESSONS:
        assert not set(trial_seeds(lesson,240,'train')) & set(trial_seeds(lesson,240,'heldout'))


def test_handover_rate_is_a_fraction_of_leap_trials():
    walker,leaper = (random_policy(adapter.SPEC,seed,hidden=8,embed=4) for seed in (1,2))
    assert 0. <= handover_rate(RoutedPolicy(walker,leaper),2,'train') <= 1.


def test_heldout_trials_use_arenas_the_training_trials_do_not():
    assert not {trial_level('train',k) for k in range(8)} & {trial_level('heldout',k) for k in range(8)}
