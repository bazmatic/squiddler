"""Training cycles every selected level equally, including level 25."""
import pytest

from experiments.run import TUNE_LEVELS
from tools import ppo_neural as ppo


def test_default_jobs_cycle_the_tune_levels():
    jobs = ppo.make_jobs(0, len(TUNE_LEVELS) * 2, 'w.npz', 6, 1.0)
    assert [job[0] for job in jobs] == list(TUNE_LEVELS) * 2


@pytest.mark.parametrize('chosen', [[25], [16], [25, 26], [3, 25, 29]])
@pytest.mark.parametrize('iteration', [0, 1, 7])
def test_jobs_cycle_exactly_the_selected_levels(chosen, iteration):
    count = 96
    jobs = ppo.make_jobs(iteration, count, 'w.npz', 6, 1.0, levels=chosen)
    assert [job[0] for job in jobs] == [chosen[n % len(chosen)]
                                       for n in range(iteration * count, (iteration + 1) * count)]



