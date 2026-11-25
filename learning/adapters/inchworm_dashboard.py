"""Physical outcomes for the inchworm curriculum."""
from learning.dashboard import Layout, Metric
from learning.inchworm_rewards import TABLE

METRICS = (Metric('stage_11_success','Leap success','Fraction of Leap episodes the policy caught the marker in, on its own (episodes an expert led are left out).',
                  section='Body',higher_is_better=True,optional=True),
           ) + tuple(Metric(key,title,description,section='Body',higher_is_better=better) for key,title,description,better in (
    ('success','Lesson success','Fraction of episodes meeting the physical lesson objective.',True),
    ('bites','Accepted bites','Actual health-reducing bites.',True),
    ('distance','Goal distance','Final body-to-goal or mouth-to-prey gap in pixels.',False),
    ('survived','Survival','Fraction still alive at episode end.',True),
))
LAYOUT = Layout(sections=('Training','Body','Policy'),strip=('success','stage_11_success','bites','distance'))
HIDDEN = ('seconds','return_total','stage','backward_distance','effort','airborne_fraction','recovered','imitation_loss','teacher_loss')
REWARD_RULES = TABLE.reward_rules()     # from the Inchworm's reward table (learning/inchworm_rewards.py)


def stage(row):
    return row.get('battle_stage')
