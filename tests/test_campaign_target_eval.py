"""The predeclared recovery gate cannot promote a candidate that misses any criterion."""
import copy
import pytest

from tools.campaign_target_eval import acceptance


def comparison():
    rows = []
    for level in (2, 3):
        for recovery in (False, True):
            rows.append(dict(level=level, target_recovery=recovery, stalled_seconds=5 if recovery else 10,
                             end='exited', cleared=True, own_kills=10, wall_crashes=2, target_switches=2))
    return rows


def test_all_gate_boundaries_allow_promotion():
    rows = comparison()
    rows[-1].update(own_kills=9.5, wall_crashes=2.5)
    result = acceptance(rows)
    assert result['passed']
    assert all(result['checks'].values())


@pytest.mark.parametrize('index,change,gate', [
    (1, {'stalled_seconds': 5.01}, 'level2_stall_reduction'),
    (3, {'end': 'scored_out'}, 'exits'),
    (3, {'cleared': False}, 'cleared_exits'),
    (3, {'own_kills': 9.49}, 'kills'),
    (3, {'wall_crashes': 2.51}, 'wall_crashes'),
])
def test_any_missed_gate_blocks_promotion(index, change, gate):
    rows = copy.deepcopy(comparison())
    rows[index].update(change)
    result = acceptance(rows)
    assert not result['passed']
    assert not result['checks'][gate]
