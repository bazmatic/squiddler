"""The Flocker team's reward settings, declared once (see learning.reward_table); shared by CLI, replay and dashboard.

Each paying term pays on one event that learning.adapters.flocker_scenario counts per bird per decision. A Flocker's
penalties are sizes, subtracted from its reward. The table's order is the order a run's manifest records them in.
"""
import math

from learning.reward_table import RewardKind, RewardTable, RewardTerm

REWARD, PENALTY, SHAPING = RewardKind.REWARD, RewardKind.PENALTY, RewardKind.SHAPING
_COEFFICIENT = 'Run-scoped Flocker reward coefficient.'

TABLE = RewardTable((
    RewardTerm('hit_reward', REWARD, 1., 'Hit reward', _COEFFICIENT, 'hits'),
    RewardTerm('kill_reward', REWARD, 5., 'Kill reward', _COEFFICIENT, 'kills'),
    RewardTerm('shot_cost', PENALTY, .05, 'Shot cost', _COEFFICIENT, 'shots'),
    RewardTerm('death_penalty', PENALTY, 5., 'Death penalty', _COEFFICIENT, 'deaths'),
    # Pixels closed on the bird's quarry over a decision, at most 20 either way.
    RewardTerm('progress_reward', REWARD, .002, 'Progress reward', _COEFFICIENT, 'closing_pixels'),
    RewardTerm('aim_progress_reward', REWARD, .2, 'Aim progress reward',
               'Credit for improving aim toward a visible target’s predicted intercept; target motion alone earns '
               'nothing.', 'aim_improvement'),
    # Sixths of a decision: closing on a teammate (-1 to 1) or keeping in line with near ones (0 to 1), per frame.
    RewardTerm('cohesion_reward', REWARD, .02, 'Cohesion reward', _COEFFICIENT, 'cohesion'),
    RewardTerm('bunch_penalty', PENALTY, .01, 'Teammate contact penalty',
               'Per decision while touching or overlapping a teammate; once per bird regardless of neighbour count.',
               'bunched_decisions'),
    RewardTerm('stuck_penalty', PENALTY, .05, 'Stuck penalty', _COEFFICIENT, 'stuck_decisions'),
    RewardTerm('perch_reward', REWARD, .1, 'Perch reward', _COEFFICIENT, 'perches'),
    # A useful volley's credit, shared by its shooters by how many of the birds able to join it did.
    RewardTerm('volley_reward', REWARD, .5, 'Volley reward', _COEFFICIENT, 'volley_credit'),
    RewardTerm('wall_collision_penalty', PENALTY, .1, 'Wall collision penalty',
               'Cost per actual wall impact, charged to the colliding bird; gentle perching costs nothing.',
               'wall_impacts'),
    RewardTerm('bunch_distance', SHAPING, 22., 'Legacy proximity distance',
               'Legacy proximity threshold in pixels; ignored when contact_only is 1.'),
    RewardTerm('contact_only', SHAPING, 1., 'Contact only (1=yes, 0=legacy proximity)',
               '1 penalizes only touching or overlapping; 0 retains the legacy proximity rule. Movement is not '
               'blocked.'),
    RewardTerm('damage_penalty', PENALTY, 2., 'Damage penalty',
               'Per hit point a Flocker loses to incoming fire (a hit does half a point at stage 5 full strength), '
               'charged to the bird hit.', 'hit_points_lost'),
    RewardTerm('threat_penalty', PENALTY, .05, 'Threat penalty',
               'Per decision a bird is in the line of fire: a hostile shot will pass within hitting distance in the '
               'next second.', 'threatened_decisions'),
), prefix='flocker_', help_prefix='Flocker: ', signed=False)

DEFAULTS = TABLE.defaults
SETTINGS = TABLE.settings


def validate(values):
    result = dict(DEFAULTS, **values)
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('Flocker reward settings must be finite and nonnegative')
    if result['contact_only'] not in (0.,1.):
        raise ValueError('contact_only must be 0 or 1')
    if result['volley_reward'] >= result['hit_reward'] and result['volley_reward'] != 0:
        raise ValueError('volley reward must remain below one accepted hit reward')
    return result


def from_args(args):
    return validate({n:getattr(args,'flocker_'+n,v) for n,v in DEFAULTS.items()})


def from_manifest(manifest):
    # Older recordings paid no impact penalty and used a fixed 35-pixel proximity threshold, rather than contact-only credit.
    return validate({'wall_collision_penalty':0., 'aim_progress_reward':0., 'bunch_distance':35., 'contact_only':0., 'damage_penalty':0., 'threat_penalty':0., **manifest.get('rewards',{})})


def recovered_settings(config):
    return {'flocker_'+n:v for n,v in from_manifest(config.get('manifest',{})).items()}
