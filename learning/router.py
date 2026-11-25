"""A rule that lets a walker checkpoint drive an Inchworm and hands it to a leaper checkpoint to leap.

`RoutedPolicy` looks like a `RecurrentPolicy` to everything that drives one (`collect`, the evaluation tools, the game's
`InchwormDriver`). Both specialists see every observation and keep their own memory in step with the world; the rule
only chooses whose action is used. It reads nothing but the observation vector, so it behaves the same in the game."""
import math
from dataclasses import dataclass, field

from learning.adapters.inchworm import FEATURES, SIGHT

GRIPS = tuple(FEATURES.index(name) for name in ('rear_grip','front_grip'))
GOAL_PRESENT, GOAL_FORWARD, GOAL_SIDE = (FEATURES.index(name) for name in ('goal_present','goal_forward','goal_side'))


@dataclass
class RouterState:
    walker: object
    leaper: object
    driver: str = 'walker'
    handovers: int = 0
    held: int = 0                 # decisions the leaper has had this hand-over
    flew: bool = False            # the worm has been in the air since this hand-over
    cooldown: int = 0
    distances: list = field(default_factory=list)      # the goal's distance over the last decisions with it in sight, gripping or not


class RoutedPolicy:
    def __init__(self, walker, leaper, reach_min=80., reach_max=200., bored_decisions=50, bored_progress=20.,
                 hold_decisions=30, cooldown_decisions=20, reach_stall=20):
        if walker.spec.features != leaper.spec.features or walker.spec.heads != leaper.spec.heads:
            raise ValueError('router specialists must share one observation and action layout')
        self.walker, self.leaper = walker, leaper
        self.spec, self.mean, self.scale = walker.spec, walker.mean, walker.scale
        self.reach_min, self.reach_max = reach_min, reach_max
        self.bored_decisions, self.bored_progress = bored_decisions, bored_progress
        self.hold_decisions, self.cooldown_decisions, self.reach_stall = hold_decisions, cooldown_decisions, reach_stall

    def initial_state(self):
        return RouterState(self.walker.initial_state(),self.leaper.initial_state())

    def choose(self, senses, state, masks=None, greedy=True, rng=None, temperature=1.0):
        driver = self._driver(senses,state)
        walked, walker_state = self.walker.choose(senses,state.walker,masks,greedy,rng,temperature)
        leapt, leaper_state = self.leaper.choose(senses,state.leaper,masks,greedy,rng,temperature)
        state.walker, state.leaper = walker_state, leaper_state
        return (leapt if driver == 'leaper' else walked), state

    def _driver(self, senses, s):
        """Update the rule's counters for this decision and settle who drives it."""
        planted = any(senses[i] > .5 for i in GRIPS)
        seen = senses[GOAL_PRESENT] > .5
        distance = SIGHT*math.hypot(senses[GOAL_FORWARD],senses[GOAL_SIDE]) if seen else None
        s.distances = (s.distances+[distance])[-self.bored_decisions:] if seen else []
        blocked = s.cooldown > 0
        s.cooldown = max(0,s.cooldown-1)
        blocked = blocked and s.cooldown > 0
        if s.driver == 'leaper':
            s.held += 1
            s.flew = s.flew or not planted
            if s.held > self.hold_decisions or (s.flew and planted):
                s.driver, s.cooldown = 'walker', self.cooldown_decisions
                return 'walker'
            return 'leaper'
        def stalled(decisions):
            recent = s.distances[-decisions:]
            return len(recent) >= decisions and recent[0]-min(recent) < self.bored_progress
        in_reach = self.reach_min <= distance <= self.reach_max if seen else False
        if planted and seen and not blocked and ((in_reach and stalled(self.reach_stall)) or stalled(self.bored_decisions)):
            s.driver, s.handovers, s.held, s.flew = 'leaper', s.handovers+1, 1, False
            s.distances = []
            return 'leaper'
        return 'walker'
