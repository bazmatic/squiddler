"""The Jelly team's reward settings, declared once (see learning.reward_table); shared by training, replay and the
dashboard. No pygame here.

Each paying term pays on one event that learning.adapters.jelly_scenario counts per Jelly per decision. A Jelly's
penalties are sizes, subtracted from its reward, and its manifest records them as negative reward terms. The run
settings at the end (a lesson, the teacher's weight, limits) are kept with the rewards but pay nothing.
"""
from learning.reward_table import RewardKind, RewardTable, RewardTerm

REWARD, PENALTY, SHAPING, SETTING = RewardKind.REWARD, RewardKind.PENALTY, RewardKind.SHAPING, RewardKind.SETTING

TABLE = RewardTable((
    RewardTerm('boredom_penalty', PENALTY, 0., 'Boredom: idle',
               'Per decision after remaining inside the configured movement radius for the full grace window, even '
               'while braking. Own catches or ship hits reset the grace period.',
               'bored_decisions', (1, 2, 3), ('rewards', 'boredom_per_decision'), rule='boredom'),
    RewardTerm('boredom_window', SHAPING, 2., 'Idle grace window',
               'Seconds of little movement before the boredom penalty begins; turning and moving detours are allowed.',
               stages=(1, 2, 3), manifest=('boredom', 'window_seconds')),
    RewardTerm('boredom_distance', SHAPING, 10., 'Idle movement radius',
               'Pixel radius of the recent position window; independent of target direction and acceleration.',
               stages=(1, 2, 3), manifest=('boredom', 'distance_pixels')),
    RewardTerm('start_lesson', SETTING, 0., 'Start lesson',
               'Navigation lesson a run begins in (0 open, 1 barrier, 2 cave); a trained start may skip ahead.',
               manifest=('training_limits', 'start_lesson')),
    RewardTerm('dagger_weight', SETTING, .5, 'Teacher weight',
               "How hard training pulls the policy toward the teacher's moves in its own states (0: no teacher).",
               manifest=('training_limits', 'dagger_weight')),
    RewardTerm('max_stage', SETTING, 3., 'Last stage',
               'The last stage a run may be promoted to (1 keeps it on solo navigation).',
               manifest=('training_limits', 'max_stage')),
    RewardTerm('no_catch_timeout', SETTING, 0., 'No-catch timeout',
               'Training rollouts only: end an episode this many seconds after its last catch (0: never).',
               manifest=('training_limits', 'no_catch_timeout_seconds')),
    RewardTerm('catch_reward', REWARD, 10., 'Catch',
               'To the Jelly that touches the target with its body; the target reappears at least 300 px from every '
               'Jelly. Stages 1-2 (and the stage-3 retention cases).',
               'catches', (1, 2, 3), ('rewards', 'catch'), rule='catch'),
    RewardTerm('team_catch_share', SHAPING, .25, 'Teammate catch share',
               "Share of the catch reward every other living Jelly gets for a teammate's catch.",
               stages=(1, 2, 3), manifest=('rewards', 'team_catch_share')),
    RewardTerm('progress_reward', REWARD, .002, 'Closing on the quarry',
               'Per pixel a Jelly closes on its quarry (the target, or the ship at stage 3) over a decision, about 0.03 '
               'at full speed; moving away costs the same.',
               'closing_pixels', (1, 2, 3), ('rewards', 'progress_per_pixel'), rule='progress'),
    RewardTerm('stuck_penalty', PENALTY, .05, 'Stuck',
               'Per decision spent stuck: pushing throughout the last second while moving less than 10 px (0.5 per '
               'second at the default).',
               'stuck_decisions', (1, 2, 3), ('rewards', 'stuck_per_decision'), rule='stuck'),
    RewardTerm('flank_reward', REWARD, .005, 'Flanking',
               'Per decision to every living Jelly, times the flank spread (0 all from one side, 1 an even ring) of the '
               'Jellies within the flank range of the quarry, when two or more are that close.',
               'flank_spread', (2, 3), ('rewards', 'stage2_flank_per_decision'), rule='flank'),
    RewardTerm('flank_range', SHAPING, 200., 'Flank range',
               'px: Jellies this close to the quarry count towards its reward spread (the chart uses a fixed 200 px '
               'benchmark).',
               stages=(2, 3), manifest=('flank', 'range')),
    RewardTerm('bunch_distance', SHAPING, 40., 'Bunching distance',
               'px between Jelly centres for the shaping penalty (the chart uses a fixed 40 px benchmark; about two '
               'body widths).',
               stages=(2, 3), manifest=('flank', 'bunch_distance')),
    RewardTerm('bunch_penalty', PENALTY, .01, 'Bunching',
               'Per decision, to each Jelly of every pair closer than the bunching distance.',
               'bunched_pairs', (2, 3), ('rewards', 'stage2_bunch_per_pair_decision'), rule='bunch'),
    RewardTerm('ship_hit_reward', REWARD, 4., 'Ship hit',
               "To the Jelly whose touch costs the training ship a hull point (none during the hull's immunity after "
               "a hit).",
               'ship_hits', (3,), ('rewards', 'stage3_ship_hit'), rule='ship_hit'),
    RewardTerm('ship_kill_reward', REWARD, 10., 'Ship kill',
               "On top of the hit reward, to the Jelly whose touch takes the ship's last hull point.",
               'ship_kills', (3,), ('rewards', 'stage3_ship_kill'), rule='ship_kill'),
    RewardTerm('team_ship_share', SHAPING, .25, 'Teammate ship share',
               "Share of a teammate's ship hit and kill rewards every other living Jelly gets.",
               stages=(3,), manifest=('rewards', 'stage3_team_ship_share')),
    RewardTerm('death_penalty', PENALTY, 5., 'Jelly killed',
               'To a Jelly the ship kills, on the decision it dies.',
               'deaths', (3,), ('rewards', 'stage3_death'), rule='death'),
    RewardTerm('team_death_share', SHAPING, .25, 'Teammate death share',
               'Share of the death penalty every surviving Jelly loses for each teammate lost.',
               stages=(3,), manifest=('rewards', 'stage3_team_death_share')),
), prefix='jelly_', help_prefix='Jelly: ', signed=False)

DEFAULTS = TABLE.defaults
# setting -> (manifest block, field, multiplier); penalties are recorded as negative reward terms.
MANIFEST_FIELDS = TABLE.manifest_fields()
SETTINGS = TABLE.settings


def from_args(args):
    return {name: getattr(args, f'jelly_{name}', default) for name, default in DEFAULTS.items()}


def from_manifest(manifest):
    """Effective values, also for recordings made before these became CLI arguments."""
    return {name: manifest.get(block, {}).get(field, default / sign) * sign
            for name, default in DEFAULTS.items() for block, field, sign in (MANIFEST_FIELDS[name],)}


def recovered_settings(config):
    """An old Jelly run used its own manifest (or the original defaults), never a later run's experiment."""
    return {f'jelly_{name}': value for name, value in from_manifest(config.get('manifest') or {}).items()}
