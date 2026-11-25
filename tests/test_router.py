"""The rule that hands leaps to a specialist checkpoint."""
from types import SimpleNamespace

import numpy as np
import pytest

from learning.adapters import inchworm as adapter
from learning.policy import random_policy
from learning.router import RoutedPolicy

FEATURES = list(adapter.FEATURES)
SIGHT = adapter.SIGHT


class Stub:
    """A specialist that always answers its own name, and counts what it was shown."""
    def __init__(self, name, spec=adapter.SPEC):
        self.name, self.spec, self.calls = name, spec, 0
        self.mean, self.scale = np.zeros(len(spec.features),np.float32), np.ones(len(spec.features),np.float32)

    def initial_state(self):
        return 0

    def choose(self, senses, state, masks=None, greedy=True, rng=None, temperature=1.0):
        self.calls += 1
        return self.name, state+1


def senses(rear=0., front=0., distance=None):
    """A feature vector: the grips as given, and the goal `distance` px dead ahead (None: no goal sensed)."""
    x = np.zeros(len(FEATURES),np.float32)
    x[FEATURES.index('rear_grip')], x[FEATURES.index('front_grip')] = rear, front
    if distance is not None:
        x[FEATURES.index('goal_present')], x[FEATURES.index('goal_forward')] = 1., min(5.,distance/SIGHT)
    return x


def run(router, steps):
    """The driver's answers to a list of feature vectors."""
    state, out = router.initial_state(), []
    for x in steps:
        action, state = router.choose(x,state)
        out.append(action)
    return out, state


def test_the_walker_drives_by_default():
    out,_ = run(RoutedPolicy(Stub('walker'),Stub('leaper')),[senses(1.,1.,40.)]*5)
    assert set(out) == {'walker'}


def test_it_hands_to_the_leaper_when_planted_and_the_goal_is_in_leaping_reach():
    out,state = run(RoutedPolicy(Stub('walker'),Stub('leaper')),[senses(0.,1.,120.)]*3)
    assert out == ['leaper']*3 and state.handovers == 1


def test_it_stays_with_the_leaper_until_the_worm_has_flown_and_landed():
    air,planted = senses(0.,0.,120.),senses(1.,1.,120.)
    out,_ = run(RoutedPolicy(Stub('walker'),Stub('leaper'),cooldown_decisions=100),[planted,air,air,planted,planted])
    assert out == ['leaper','leaper','leaper','walker','walker']


def test_it_hands_back_after_the_hold_even_if_the_worm_never_lands():
    out,_ = run(RoutedPolicy(Stub('walker'),Stub('leaper'),hold_decisions=30,cooldown_decisions=100),[senses(0.,1.,120.)]*34)
    assert out[:30] == ['leaper']*30 and out[30:] == ['walker']*4


def test_it_does_not_hand_over_again_during_the_cooldown():
    steps = [senses(0.,1.,120.)]*2+[senses(1.,1.,120.)]*5
    out,_ = run(RoutedPolicy(Stub('walker'),Stub('leaper'),hold_decisions=2,cooldown_decisions=4),steps)
    assert out == ['leaper']*2+['walker']*4+['leaper']


def test_a_worm_gripping_without_getting_closer_is_bored_and_leaps_from_any_visible_distance():
    router = RoutedPolicy(Stub('walker'),Stub('leaper'),bored_decisions=50,bored_progress=20.)
    out,state = run(router,[senses(1.,1.,250.)]*60)
    assert out[0] == 'walker' and out[-1] == 'leaper' and state.handovers == 1


def test_a_worm_that_is_closing_in_is_not_bored():
    steps = [senses(1.,1.,290.-3.*k) for k in range(60)]        # 3 px closer every decision
    out,_ = run(RoutedPolicy(Stub('walker'),Stub('leaper'),reach_min=1.,reach_max=2.),steps)
    assert set(out) == {'walker'}


def test_both_grips_free_never_starts_a_handover():
    out,_ = run(RoutedPolicy(Stub('walker'),Stub('leaper')),[senses(0.,0.,120.)]*60)
    assert set(out) == {'walker'}


def test_no_goal_never_hands_over_or_gets_bored():
    out,_ = run(RoutedPolicy(Stub('walker'),Stub('leaper')),[senses(1.,1.,None)]*80)
    assert set(out) == {'walker'}


def test_both_specialists_see_every_observation():
    walker,leaper = Stub('walker'),Stub('leaper')
    run(RoutedPolicy(walker,leaper),[senses(1.,1.,120.)]*7)
    assert walker.calls == leaper.calls == 7


def test_mismatched_specialists_are_rejected():
    other = adapter.SPEC.__class__('inchworm',adapter.FEATURES[:-1],adapter.HEADS,version=1,continuous=adapter.MOTOR_NAMES)
    with pytest.raises(ValueError):
        RoutedPolicy(Stub('walker'),Stub('leaper',spec=other))


def test_a_new_initial_state_forgets_the_counters():
    router = RoutedPolicy(Stub('walker'),Stub('leaper'),bored_decisions=5)
    _,state = run(router,[senses(1.,1.,250.)]*8)
    assert state.handovers == 1
    assert router.initial_state().handovers == 0 and router.initial_state().driver == 'walker'


def test_the_router_is_deterministic_for_a_seed():
    walker,leaper = (random_policy(adapter.SPEC,seed,hidden=8,embed=4) for seed in (1,2))
    steps = [senses(1.,1.,120.+k) for k in range(20)]
    def actions():
        router,state,rng,out = RoutedPolicy(walker,leaper),None,np.random.default_rng(3),[]
        state = router.initial_state()
        for x in steps:
            a,state = router.choose(x,state,greedy=False,rng=rng); out.append(a)
        return out
    assert actions() == actions()
