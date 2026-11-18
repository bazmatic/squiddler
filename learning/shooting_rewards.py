"""The shared hunter shooting check, using direct-sight enemy bearings in either observation layout."""
from functools import lru_cache

import numpy as np

AIM_DEGREES = 20.0


@lru_cache(maxsize=None)
def _enemy_indices(feature_names):
    slots = [name[:-len('_present')] for name in feature_names
             if name.startswith('enemy') and name.endswith('_present')
             and name[len('enemy'):-len('_present')].isdigit()]
    return tuple(tuple(feature_names.index(f'{slot}_{part}') for part in ('present', 'sin', 'cos'))
                 for slot in slots)


def unaimed_shots(senses, shots, feature_names):
    """Count bullets per decision with no directly seen enemy within 20 degrees of the nose.

    This is the original hunter's bearing test, not a predicted intercept or hit test. Every bullet in a decision
    interval uses that decision's sight snapshot. Remembered enemies do not count as direct sight.
    """
    names = tuple(feature_names)
    observations = np.asarray(senses).reshape(-1, len(names))
    aimed = np.zeros(len(observations), dtype=bool)
    for present, sin, cos in _enemy_indices(names):
        off = np.degrees(np.abs(np.arctan2(observations[:, sin], observations[:, cos])))
        aimed |= (observations[:, present] == 1) & (off <= AIM_DEGREES)
    return np.where(aimed, 0.0, np.asarray(shots, dtype=np.float64))
