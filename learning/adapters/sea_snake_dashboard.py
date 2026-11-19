"""Hunting outcomes for the sea snake curriculum."""
from learning.dashboard import Layout, Metric
from learning.sea_snake_rewards import TABLE, recovered_settings  # noqa: F401 (recovered_settings: see learning.dashboard)

METRICS = tuple(Metric(key, title, description, section='Body', higher_is_better=better)
                for key, title, description, better in (
    ('success', 'Lesson success', 'Fraction of episodes that sting the target.', True),
    ('sting_speed', 'Sting speed', 'Time left when it stings, as a fraction of the episode (0: no sting).', True),
    ('distance', 'Head distance', 'Final head-to-target gap in snake radii, by way of any wall gaps.', False),
    ('effort', 'Muscle effort', 'Mean bend the muscles hold, from limp (0) to fully bent (1).', False),
    ('wall_fraction', 'Against walls', 'Fraction of frames with the body pressed against a wall.', False),
    ('resets', 'Solver resets', 'Times per episode the physics blew up and the snake was laid out again.', False),
    ('wriggle', 'Wriggle', 'How snake-like it swims, 0-1: the geometric mean of bend size, wave, rhythm and swim, '
                           'times body spread. Deployment needs .65 in every lesson.', True),
    ('wriggle_curvature', 'Bend size', 'Bends neither rigid nor folded, and the body not coiled.', True),
    ('wriggle_spread', 'Body spread', 'How much of the body swings: low when only the tail flaps.', True),
    ('wriggle_wave', 'Head-to-tail wave', 'Bends travel from head to tail.', True),
    ('wriggle_rhythm', 'Rhythm', 'A plausible undulation rate (0.5-2.5 per second) without twitching.', True),
    ('wriggle_swim', 'Forward swim', 'Moves forward along its heading (1 radius per second scores full).', True),
))
LAYOUT = Layout(sections=('Training', 'Body', 'Policy'), strip=('success', 'wriggle', 'distance'))
HIDDEN = ('seconds', 'return_total', 'stage', 'survived')
REWARD_RULES = TABLE.reward_rules()     # from the Sea Snake's reward table (learning/sea_snake_rewards.py)


def stage(row):
    return row.get('battle_stage')
