"""The Campaign Hunter's reward points, declared once (see learning.reward_table); no pygame here.

Each term pays on one event that experiments.campaign_rollout counts per decision. The exit and score-out events are
counted once, on the episode's last decision.
"""
from learning.reward_table import RewardTable, RewardTerm

TABLE = RewardTable((   # starting values, tuned against the Demo bar
    RewardTerm('own_kill', 10.0, 'Own kill', 'points per enemy the ship destroys itself', 'own_kills'),
    RewardTerm('pickup', 8.0, 'Pickup', 'points per crystal the ship collects', 'pickups'),
    RewardTerm('beacon', 5.0, 'Beacon', 'points for fetching an Ally beacon', 'beacon'),
    RewardTerm('exit', 100.0, 'Exit', 'points for leaving through the exit', 'exited'),
    # On top of `exit` when no enemy is left, whoever killed them.
    RewardTerm('cleared_exit', 100.0, 'Cleared exit', 'more points for leaving through the exit with no enemy left',
               'cleared_exit'),
    RewardTerm('score_out', -100.0, 'Score out', 'points when the score runs out (negative: a penalty)', 'scored_out'),
    RewardTerm('wall_crash', -1.0, 'Wall crash', 'points per wall crash (negative: a penalty)', 'wall_crashes'),
    RewardTerm('compass_following', 0.0, 'Compass following',
               "points per decision flying with the enemy compass (the exit's once none remain)", 'following'),
    # Ground along the route, in maze widths; moving away costs the same.
    RewardTerm('hunt_progress', 20.0, 'Hunt progress',
               'points per maze width closer to the nearest enemy by route (moving away costs the same)',
               'hunt_progress'),
    RewardTerm('exit_progress', 20.0, 'Exit progress',
               'points per maze width closer to the open exit by route (moving away costs the same)', 'exit_progress'),
    RewardTerm('idle', -0.05, 'Idling',
               'points per decision spent coasting at a crawl with no thrust (negative: a penalty)', 'idle'),
    # Fuel is score, so coasting is worth learning.
    RewardTerm('thrust', -0.02, 'Thrusting', 'points per decision with the thruster on (negative: a penalty)',
               'thrusting'),
    # Ammunition is score, so each shot is worth aiming.
    RewardTerm('shot', 0.0, 'Shots fired', 'points per bullet fired (negative: a penalty)', 'shots'),
), help_prefix='Campaign Hunter: ')

CampaignShaping = TABLE.shaping_class('CampaignShaping', 'Points per event, one field per reward term.')
DEFAULTS = TABLE.defaults
HELP = {t.key: t.help for t in TABLE.terms}
SETTINGS = TABLE.settings            # the trainer's flags: --own-kill ... --shot
