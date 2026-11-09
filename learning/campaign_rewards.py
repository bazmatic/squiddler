"""The Campaign Hunter's reward points, declared once (see learning.reward_table); no pygame here.

Each term pays on one event that experiments.campaign_rollout counts per decision. The exit and score-out events are
counted once, on the episode's last decision.
"""
from learning.reward_table import RewardKind, RewardTable, RewardTerm

REWARD, PENALTY = RewardKind.REWARD, RewardKind.PENALTY

TABLE = RewardTable((   # starting values, tuned against the Demo bar
    RewardTerm('own_kill', REWARD, 10.0, 'Own kill', 'points per enemy the ship destroys itself', 'own_kills'),
    RewardTerm('pickup', REWARD, 8.0, 'Pickup', 'points per crystal the ship collects', 'pickups'),
    RewardTerm('beacon', REWARD, 5.0, 'Beacon', 'points for fetching an Ally beacon', 'beacon'),
    RewardTerm('exit', REWARD, 100.0, 'Exit', 'points for leaving through the exit', 'exited'),
    # On top of `exit` when no enemy is left, whoever killed them.
    RewardTerm('cleared_exit', REWARD, 100.0, 'Cleared exit',
               'more points for leaving through the exit with no enemy left', 'cleared_exit'),
    RewardTerm('score_out', PENALTY, -100.0, 'Score out', 'points when the score runs out', 'scored_out'),
    RewardTerm('wall_crash', PENALTY, -1.0, 'Wall crash', 'points per wall crash', 'wall_crashes'),
    RewardTerm('compass_following', REWARD, 0.0, 'Compass following',
               "points per decision flying with the enemy compass (the exit's once none remain)", 'following'),
    # Ground along the route, in maze widths; moving away costs the same.
    RewardTerm('hunt_progress', REWARD, 20.0, 'Hunt progress',
               'points per maze width closer to the nearest enemy by route (moving away costs the same)',
               'hunt_progress'),
    RewardTerm('exit_progress', REWARD, 20.0, 'Exit progress',
               'points per maze width closer to the open exit by route (moving away costs the same)', 'exit_progress'),
    RewardTerm('idle', PENALTY, -0.05, 'Idling', 'points per decision spent coasting at a crawl with no thrust', 'idle'),
    # Fuel is score, so coasting is worth learning.
    RewardTerm('thrust', PENALTY, -0.02, 'Thrusting', 'points per decision with the thruster on', 'thrusting'),
    # Ammunition is score, so each shot is worth aiming.
    RewardTerm('shot', PENALTY, 0.0, 'Shots fired', 'points per bullet fired', 'shots'),
    # Pivoting from side to side is a shake, not a turn.
    RewardTerm('turn_change', PENALTY, 0.0, 'Direction changes',
               'points per decision that turns the opposite way from the one before', 'turn_changes'),
), help_prefix='Campaign Hunter: ')

CampaignShaping = TABLE.shaping_class('CampaignShaping', 'Points per event, one field per reward term.')
DEFAULTS = TABLE.defaults
HELP = {t.key: TABLE.described(t) for t in TABLE.terms}   # as the trainer's flags give it
SETTINGS = TABLE.settings            # the trainer's flags: --own-kill ... --turn-change
