"""The hunter's (the Viber's) training rewards, declared once (see learning.reward_table); no pygame here.

Each paying term pays on one event that experiments.rollout counts per decision. A hunter's penalties are sizes,
subtracted from its reward. The kill reward and the idle cost are fixed, not run settings; the wall and enemy-hit
settings scale the lab score's own weights. A run records the terms as its rewards summary rows, in this order and
under these names, and its arguments under these keys.
"""
from experiments.score import Weights
from learning.reward_table import RewardKind, RewardTable, RewardTerm, RowNames

REWARD, PENALTY, SHAPING = RewardKind.REWARD, RewardKind.PENALTY, RewardKind.SHAPING

KILL_REWARD_TOTAL = 100.0       # shared across every enemy that existed in the episode
IDLE_AFTER_SECONDS = 20.0       # experiments.harness counts idle time only after this long without a kill
_WEIGHTS = Weights()

TABLE = RewardTable((
    RewardTerm('kills', REWARD, KILL_REWARD_TOTAL, 'Kills', 'points shared across all the enemies in an episode',
               'kill_share', fixed=True),
    RewardTerm('powerup_reward', REWARD, 0., 'Powerup collected', 'points each', 'powerups', rule='powerup'),
    RewardTerm('upgrade_bonus', REWARD, 0., 'Gun upgraded', 'points per second with a hunter power boost',
               'boosted_seconds', rule='upgrade'),
    RewardTerm('wall_scale', PENALTY, 1., 'Wall impact', 'points per full-speed hit (scales with the speed lost)',
               'wall_impulse', rule='wall', unit=_WEIGHTS.quotient),
    RewardTerm('hit_scale', PENALTY, 1., 'Enemy hit', 'points per enemy collision or projectile hit', 'hits_taken',
               rule='enemy_hit',
               unit=_WEIGHTS.hit),
    RewardTerm('idle', PENALTY, _WEIGHTS.idle / 60.0, 'Idle',
               f'points per second beyond {IDLE_AFTER_SECONDS:.0f} s without a kill, until the field is clear',
               'idle_seconds', fixed=True),
    RewardTerm('speed_penalty', PENALTY, 0., 'Flying below top speed',
               'points per second at a standstill, in proportion to the shortfall', 'shortfall_seconds', rule='speed'),
    RewardTerm('still_penalty', PENALTY, 0., 'Crawling under the still speed',
               'points per second below the still speed', 'still_seconds', rule='still'),
    RewardTerm('still_speed', SHAPING, .15, 'Still speed (share of top speed)',
               'below this share of top speed counts as near-motionless'),
    RewardTerm('turn_change_penalty', PENALTY, 0., 'Starting or reversing a turn', 'points each', 'turn_starts',
               rule='turn_change'),
    RewardTerm('reversal_penalty', PENALTY, 0., 'Reversing a turn directly', 'extra points each, left to right or back',
               'turn_reversals', rule='reversal'),
    RewardTerm('death_penalty', PENALTY, 0., 'Death',
               'points when the hunter is destroyed (it survives as many hits as its health: see Reward details)',
               'deaths', rule='death'),
    RewardTerm('rules_bounty', REWARD, 50., 'Rules hunter destroyed', 'points, on top of its share of the kill reward',
               'rules_hunters_beaten'),
    RewardTerm('compass_reward', REWARD, 0., 'Following the enemy compass',
               'points per maze width the route to the nearest enemy shortens while none is in sight',
               'compass_progress', rule='compass'),
    RewardTerm('hit_reward', REWARD, 0., 'Hit on an enemy', 'points per shot that reaches an enemy', 'hits',
               rule='hit'),
    RewardTerm('turning_penalty', PENALTY, 0., 'Turning', 'points per second spent turning', 'turning_seconds',
               rule='turning'),
    # Runs before the rename recorded the unaimed shot price as "Shot with no enemy in sight".
    RewardTerm('shot_price', PENALTY, 0., 'Unaimed shot', 'points each, unless within 20° of an enemy in sight',
               'unaimed_shots', rule='unaimed_shot', recorded_as=('Shot with no enemy in sight',)),
    RewardTerm('boredom_penalty', PENALTY, 0., 'Boredom',
               'points per second with no enemy in sight in a cell visited within the boredom window', 'bored_seconds',
               rule='boredom'),
    RewardTerm('boredom_window', SHAPING, 10., 'Boredom window (s)',
               'seconds before a visited maze cell counts as new again'),
), help_prefix='Hunter: ', signed=False, rule_defaults=False, row_names=RowNames.TITLE)

DEFAULTS = TABLE.defaults
SETTINGS = TABLE.settings
