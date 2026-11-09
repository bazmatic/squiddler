"""The Inchworm's reward settings, declared once (see learning.reward_table); potential progress is signed to prevent
oscillation farming.

Each paying term pays on one event that learning.adapters.inchworm_scenario counts per decision. An Inchworm's
penalties are sizes, subtracted from its reward. The table's order is the order a run's manifest records them in.
"""
import math

from learning.reward_table import RewardKind, RewardTable, RewardTerm

REWARD, PENALTY, SHAPING = RewardKind.REWARD, RewardKind.PENALTY, RewardKind.SHAPING
_COEFFICIENT = 'Run-scoped inchworm reward coefficient; its rate and events are defined in learning/inchworm_rewards.py.'

TABLE = RewardTable((
    RewardTerm('progress_reward', REWARD, .04, 'Progress reward',
               'Head-first approach earns progress; retreat pays in full. Rear/curl drills use signed mouth-to-marker '
               'progress.', 'progress_pixels'),
    RewardTerm('bite_reward', REWARD, 2., 'Bite reward', _COEFFICIENT, 'bites'),
    RewardTerm('success_reward', REWARD, 4., 'Success reward', _COEFFICIENT, 'successes'),
    RewardTerm('death_penalty', PENALTY, 5., 'Death penalty', _COEFFICIENT, 'deaths'),
    # Per second of the decision's frames, in proportion to the share of bending torque used.
    RewardTerm('effort_cost', PENALTY, .03, 'Effort cost', _COEFFICIENT, 'effort_seconds'),
    # Per second with neither grip, unless stunned.
    RewardTerm('airborne_cost', PENALTY, .08, 'Airborne cost', _COEFFICIENT, 'airborne_seconds'),
    RewardTerm('time_cost', PENALTY, .002, 'Time cost', _COEFFICIENT, 'seconds'),
    # The mean squared change of the applied controls from the last decision's.
    RewardTerm('action_change_cost', PENALTY, .01, 'Action change cost', _COEFFICIENT, 'action_change'),
    RewardTerm('backward_cost', PENALTY, .04, 'Backward cost',
               'Net tail-first travel of the body centre along the surface, after crediting up to 30 px of forward '
               'travel; excludes retained rear anchors and forced stun motion.', 'backward_pixels'),
    RewardTerm('turn_reward', REWARD, 3., 'Turn reward',
               'About-face only: change in (facing the goal, clamped at 0) × (body stretched out); finishing the turn '
               'pays this once.', 'turn_progress'),
    RewardTerm('dagger_weight', SHAPING, 1., 'Dagger weight',
               'Weight pulling the policy toward the scripted controller in the states it reaches during About-face '
               '(not a reward).'),
    RewardTerm('speed_bonus', REWARD, 4., 'Speed bonus',
               'Crawl only: extra reward on arrival, scaled by the fraction of the time limit still left, so a faster '
               'crawl pays more.', 'time_left_on_arrival'),
    RewardTerm('crawl_time_cost', PENALTY, .05, 'Crawl time cost',
               'Crawl only: extra penalty per second on top of the time cost, so a slow crawl loses reward the whole '
               'way, not just at arrival.', 'crawl_seconds'),
    RewardTerm('about_face_time_cost', PENALTY, .05, 'About face time cost',
               'About-face only: extra penalty per second on top of the time cost, so turning and getting going '
               'quickly pays.', 'about_face_seconds'),
    RewardTerm('end_pitch_cost', PENALTY, .3, 'End pitch cost',
               'Walking only (Crawl, and About-face once it faces the goal): penalty per second when the tail or head '
               'link is more than 15 degrees off parallel to the ground, scaled up to a right angle.',
               'end_pitch_seconds'),
), prefix='inchworm_', help_prefix='Inchworm: ', signed=False)

DEFAULTS = TABLE.defaults
SETTINGS = TABLE.settings


def validate(values):
    if set(values)-set(DEFAULTS):
        raise ValueError('unknown inchworm reward setting')
    result = dict(DEFAULTS, **values)
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('inchworm rewards must be finite and nonnegative')
    return result


def from_args(args):
    return validate({n:getattr(args,'inchworm_'+n,v) for n,v in DEFAULTS.items()})


def recovered_settings(config):
    return {'inchworm_'+n:v for n,v in validate(config.get('manifest',{}).get('rewards',{})).items()}
