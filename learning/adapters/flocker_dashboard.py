"""Flocker outcome, coordination and rest metrics."""
from learning.dashboard import Layout, Metric
from learning.flocker_rewards import TABLE, recovered_settings  # noqa: F401 (recovered_settings: see learning.dashboard)

METRICS = tuple(Metric(key,title,description,unit='%' if key in ('perch_success','accuracy','aim_accuracy') else '',section=section,higher_is_better=better) for key,title,description,section,better in (
    ('hits_per_minute','Hits / minute','Accepted projectile damage per minute.','Combat',True),
    ('accuracy','Damage / shot','Shots that reduce health divided by emitted shots; damage immunity can block otherwise accurate hits.','Combat',True),
    ('aim_accuracy','Aim accuracy','Shots that physically land divided by emitted shots, including hits blocked by damage immunity.','Combat',True),
    ('survival_fraction','Survival','Fraction of Flockers alive at episode end.','Combat',True),
    ('useful_volleys_per_minute','Useful volleys / minute','Synchronized same-target launch groups that hit.','Coordination',True),
    ('participation','Volley participation','Share of eligible nearby teammates joining a volley.','Coordination',True),
    ('median_launch_spread','Launch spread','Median frames between first and last shot; 7 means no volleys.','Coordination',False),
    ('cohesion','Cohesion','Share of Flocker time with a teammate within 100 pixels.','Flight',True),
    ('alignment','Velocity alignment','Time-weighted agreement with nearby teammates’ travel directions; stationary birds earn zero.','Flight',True),
    ('bunched_fraction','Teammate contact','Share of time touching or overlapping a teammate; legacy proximity runs retain their recorded distance rule.','Flight',False),
    ('hard_landings','Wall impacts','Hard wall, gate and boundary collisions; gentle perches are excluded.','Flight',False),
    ('wall_collision_cost','Wall collision cost','Total reward deducted from the birds that collided.','Flight',False),
    ('stuck_fraction','Stuck','Share of time thrusting without meaningful movement.','Flight',False),
    ('mean_throttle','Mean throttle','Applied continuous throttle, from zero to one.','Flight',None),
    ('fractional_throttle_fraction','Fractional throttle','Time applying throttle between 0.05 and 0.95.','Flight',None),
    ('perches','Gentle perches','Successful physical wall attachments.','Perching',True),
    ('perch_success','Perch success (%)','Successful attachments divided by attempts in training episodes. Stage 2 requires 90% on held-out validation trials.','Perching',True),
    ('perched_fraction','Time perched','Share of Flocker time attached to a wall.','Perching',None),
))
LAYOUT = Layout(sections=('Training','Combat','Coordination','Flight','Perching','Policy'),
                strip=('hits_per_minute','useful_volleys_per_minute','survival_fraction'))
HIDDEN = ('seconds','episodes','shots','hits','landed_hits','non_damaging_hits','volleys','kills')
REWARD_RULES = TABLE.reward_rules()     # from the Flocker's reward table (learning/flocker_rewards.py)


def stage(row):
    return row.get('battle_stage')
