"""The Campaign Hunter's reward points, shared by its episodes and the trainer's settings (no pygame here)."""
from dataclasses import dataclass, fields


@dataclass
class CampaignShaping:
    """Points per event (starting values; tuned against the Demo bar)."""
    own_kill: float = 10.0
    pickup: float = 3.0
    beacon: float = 5.0
    exit: float = 100.0
    cleared_exit: float = 100.0          # on top of `exit` when no enemy is left, whoever killed them
    score_out: float = -100.0
    wall_crash: float = -1.0
    compass_following: float = 0.02      # per decision flying with the enemy compass (the exit's once none remain)


DEFAULTS = {field.name: field.default for field in fields(CampaignShaping)}
HELP = {'own_kill': 'points per enemy the ship destroys itself',
        'pickup': 'points per crystal the ship collects',
        'beacon': 'points for fetching an Ally beacon',
        'exit': 'points for leaving through the exit',
        'cleared_exit': 'more points for leaving through the exit with no enemy left',
        'score_out': 'points when the score runs out (negative: a penalty)',
        'wall_crash': 'points per wall crash (negative: a penalty)',
        'compass_following': "points per decision flying with the enemy compass (the exit's once none remain)"}
SETTINGS = tuple(DEFAULTS)            # the trainer's flags: --own-kill ... --compass-following
