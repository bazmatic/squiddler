"""Physical outcomes for the inchworm curriculum."""
from learning.dashboard import Layout, Metric
from learning.inchworm_rewards import TABLE, recovered_settings  # noqa: F401 (recovered_settings: see learning.dashboard)

METRICS = tuple(Metric(key,title,description,section='Body',higher_is_better=better) for key,title,description,better in (
    ('success','Lesson success','Fraction of episodes meeting the physical lesson objective.',True),
    ('bites','Accepted bites','Actual health-reducing bites.',True),
    ('distance','Goal distance','Final body-to-goal or mouth-to-prey gap in pixels.',False),
    ('backward_distance','Backward travel','Tail-first rear-segment travel in pixels; anchored bends and stun motion excluded.',False),
    ('effort','Muscle effort','Mean fraction of available bending torque.',False),
    ('airborne_fraction','Detached time','Fraction of frames without either grip.',False),
    ('recovered','Recovered grip','Fraction with at least 30 planted frames after the scheduled shove.',True),
    ('survived','Survival','Fraction still alive at episode end.',True),
))
LAYOUT = Layout(sections=('Training','Body','Policy'),strip=('success','bites','distance'))
HIDDEN = ('seconds','return_total','stage')
REWARD_RULES = TABLE.reward_rules()     # from the Inchworm's reward table (learning/inchworm_rewards.py)


def stage(row):
    return row.get('battle_stage')
