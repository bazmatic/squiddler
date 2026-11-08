"""The Campaign Hunter's reward points, shared by its episodes and the trainer's settings (no pygame here)."""
from dataclasses import dataclass, fields


@dataclass
class CampaignShaping:
    """Points per event (starting values; tuned against the Demo bar)."""
    own_kill: float = 10.0
    pickup: float = 8.0
    beacon: float = 5.0
    exit: float = 100.0
    cleared_exit: float = 100.0          # on top of `exit` when no enemy is left, whoever killed them
    score_out: float = -100.0
    wall_crash: float = -1.0
    compass_following: float = 0.0       # per decision flying with the enemy compass (the exit's once none remain)
    hunt_progress: float = 20.0          # per maze width the route to the nearest enemy gets shorter (longer: the same, taken)
    exit_progress: float = 20.0          # per maze width the route to the open exit gets shorter (longer: the same, taken)
    idle: float = -0.05                  # per decision spent coasting at a crawl without thrust


DEFAULTS = {field.name: field.default for field in fields(CampaignShaping)}
HELP = {'own_kill': 'points per enemy the ship destroys itself',
        'pickup': 'points per crystal the ship collects',
        'beacon': 'points for fetching an Ally beacon',
        'exit': 'points for leaving through the exit',
        'cleared_exit': 'more points for leaving through the exit with no enemy left',
        'score_out': 'points when the score runs out (negative: a penalty)',
        'wall_crash': 'points per wall crash (negative: a penalty)',
        'compass_following': "points per decision flying with the enemy compass (the exit's once none remain)",
        'hunt_progress': 'points per maze width closer to the nearest enemy by route (moving away costs the same)',
        'exit_progress': 'points per maze width closer to the open exit by route (moving away costs the same)',
        'idle': 'points per decision spent coasting at a crawl with no thrust (negative: a penalty)'}
SETTINGS = tuple(DEFAULTS)            # the trainer's flags: --own-kill ... --compass-following
