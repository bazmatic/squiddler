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
    RewardTerm('wall_crash', PENALTY, -1.0, 'Wall impact',
               'points per ship top-speed unit of collision-induced velocity change; '
               'glancing and slow contacts cost less, and repeated impacts accumulate', 'wall_impulse'),
    RewardTerm('compass_following', REWARD, 0.0, 'Compass following',
               "points per decision flying with the enemy compass (the exit's once none remain)", 'following'),
    # Ground along the route, in maze widths; moving away costs the same.
    RewardTerm('hunt_progress', REWARD, 20.0, 'Hunt progress',
               'points per maze width the ship advances toward its preceding enemy goal by route; '
               'enemy movement earns nothing and retreat costs the same',
               'hunt_progress'),
    RewardTerm('exit_progress', REWARD, 20.0, 'Exit progress',
               'points per maze width the ship advances toward the open exit by route '
               '(moving away costs the same)', 'exit_progress'),
    RewardTerm('unseen_progress', REWARD, 0.0, 'Unseen enemy pursuit',
               'extra points per maze width the ship advances toward an enemy while none is visible; '
               'retreat costs the same, and enemy movement earns nothing', 'unseen_progress'),
    RewardTerm('search_contact', REWARD, 0.0, 'Pursuit lesson contact',
               'points for first seeing an enemy in a pursuit lesson that began with none visible; '
               'never paid in normal campaign episodes', 'search_contacts'),
    RewardTerm('idle', PENALTY, -0.05, 'Idling', 'points per decision spent coasting at a crawl with no thrust', 'idle'),
    # Fuel is score, so coasting is worth learning.
    RewardTerm('thrust', PENALTY, -0.02, 'Thrusting', 'points per decision with the thruster on', 'thrusting'),
    # Ammunition is score, so each shot is worth aiming.
    RewardTerm('shot', PENALTY, 0.0, 'Shots fired',
               'points per shot; graded-shooting runs waive this cost for rough aim toward a seen enemy', 'shots'),
    RewardTerm('enemy_hit', REWARD, 0.0, 'Damaging hit on an enemy',
               'points per player projectile hit that reduces enemy health; absorbed shots and ally hits earn nothing',
               'enemy_hits'),
    RewardTerm('unaimed_shot', PENALTY, 0.0, 'Unaimed shot',
               'extra points per shot outside the rough aim window; graded runs use 30 degrees around current '
               'or intercept bearing at firing time, historical runs use the 20-degree decision snapshot',
               'unaimed_shots'),
    RewardTerm('missed_opportunity', PENALTY, 0.0, 'Declined firing opportunity',
               'points per weapon firing interval spent not firing when ready and roughly aimed toward a seen '
               'enemy with a clear reachable intercept; historical runs use the strict direct-hit check',
               'missed_opportunities'),
    RewardTerm('shot_accuracy', REWARD, .05, 'Shot accuracy',
               'maximum aim-only points per volley toward a clear predicted intercept; new runs square the '
               'alignment score so near misses earn very little. Actual damaging hits pay separately.', 'accurate_shots'),
    RewardTerm('breach_block', REWARD, 0.0, 'Necessary block destroyed',
               'points per player-destroyed block on a planned breach route while no enemy has an open route; '
               'partial damage and unrelated blocks earn nothing', 'breach_blocks'),
    # Pivoting from side to side is a shake, not a turn; a single deliberate reversal is free.
    RewardTerm('turn_change', PENALTY, 0.0, 'Quick reversals',
               'points for reversing the last nonzero turn direction when the previous reversal was only a moment '
               'before (about half a second); starting or stopping a turn and an unhurried reversal are free',
               'turn_changes'),
), help_prefix='Campaign Hunter: ')

CampaignShaping = TABLE.shaping_class('CampaignShaping', 'Points per event, one field per reward term.')
DEFAULTS = TABLE.defaults
HELP = {t.key: TABLE.described(t) for t in TABLE.terms}   # as the trainer's flags give it
SETTINGS = TABLE.settings            # the trainer's flags: --own-kill ... --turn-change
