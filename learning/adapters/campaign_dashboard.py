"""The Campaign Hunter's dashboard: the keys learning.adapters.campaign_hunter.summary() adds to each train.log row
(tools/ppo_neural.py writes the PPO stats and mean_return beside them), and its reward points from
learning.campaign_rewards."""
from learning.campaign_rewards import DEFAULTS, HELP
from learning.dashboard import Layout, Metric, RewardRule, number

METRICS = (
    Metric('exited', 'Episodes exited',
           'Higher is better. Share of episodes in which the ship leaves the level through the exit.',
           section='Outcome', higher_is_better=True, short='Exited'),
    Metric('cleared_exit', 'Episodes exited with every enemy dead',
           'Higher is better. Share of episodes in which the ship leaves through the exit with no enemy left.',
           section='Outcome', higher_is_better=True, short='Cleared exit'),
    Metric('scored_out', 'Episodes scored out',
           'Lower is better. Share of episodes that end because the score runs out.',
           section='Outcome', higher_is_better=False, short='Scored out'),
    Metric('own_kills', 'Own kills per episode',
           'Higher is better. Enemies the ship destroys itself, on average per episode.',
           section='Outcome', higher_is_better=True, short='Own kills', digits=1),
    Metric('wall_crashes', 'Wall strikes per episode',
           'Lower is better. Times the ship strikes a wall, on average per episode.',
           section='Flying', higher_is_better=False, short='Wall strikes', digits=1, optional=True),
    Metric('shots', 'Bullets fired per episode',
           'Bullets the ship fires, on average per episode.',
           section='Flying', short='Shots', digits=1, optional=True),
    Metric('thrust', 'Thruster use',
           'Share of the pilot\'s decisions with the thruster on. 1 means it never coasts.',
           section='Flying', short='Thrust', optional=True),
)

LAYOUT = Layout(sections=('Training', 'Outcome', 'Flying', 'Policy'), strip=('exited', 'cleared_exit', 'own_kills'))

_TITLES = dict(own_kill='Own kill', pickup='Pickup', beacon='Beacon', exit='Exit', cleared_exit='Cleared exit',
               score_out='Score out', wall_crash='Wall crash', compass_following='Compass following',
               hunt_progress='Hunt progress', exit_progress='Exit progress', idle='Idling', thrust='Thrusting', shot='Shots fired',
               turn_change='Direction changes')


def _kind(key):
    return 'penalty' if DEFAULTS[key] < 0 or '(negative' in HELP[key] else 'reward'


# A penalty's value is its size: the page draws it with a minus sign. Help text drops the "(negative: a penalty)" note.
REWARD_RULES = tuple(RewardRule(key, _TITLES[key], HELP[key].split(' (negative')[0], _kind(key), (), key,
                                abs(DEFAULTS[key])) for key in DEFAULTS)


def recorded(config):
    """The points config.json's args record, with a penalty's sign flipped so the page shows it as a penalty."""
    args = (config or {}).get('args') or {}
    return {r.key: (-args[r.setting] if r.kind == 'penalty' else args[r.setting]) for r in REWARD_RULES
            if isinstance(args.get(r.setting), (int, float)) and not isinstance(args.get(r.setting), bool)}


def cards(config):
    e = config.get('episode') or {}
    if not e:
        return []
    text = (f"Whole Demo levels of up to {number(e.get('horizon_seconds'))} s, ending at the exit or when the score runs "
            f"out. {number(e.get('episodes_per_iteration'))} episodes per iteration.")
    if e.get('levels'):
        text += f" Levels: {','.join(str(level) for level in e['levels'])}."
    return [dict(title='Episodes', rows=[], note=text)]
