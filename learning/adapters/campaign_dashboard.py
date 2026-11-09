"""The Campaign Hunter's dashboard: the keys learning.adapters.campaign_hunter.summary() adds to each train.log row
(tools/ppo_neural.py writes the PPO stats and mean_return beside them), and its reward points from
the reward table in learning.campaign_rewards."""
from learning.campaign_rewards import TABLE
from learning.reward_table import RewardKind
from learning.dashboard import Layout, Metric, number

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

REWARD_RULES = TABLE.reward_rules()


def recorded(config):
    """The points config.json's args record, with a penalty's sign flipped so the page shows it as a penalty."""
    args = (config or {}).get('args') or {}
    values = {t.key: args[t.key] for t in TABLE.terms
              if isinstance(args.get(t.key), (int, float)) and not isinstance(args.get(t.key), bool)}
    return {key: -value if TABLE.by_key[key].kind is RewardKind.PENALTY else value for key, value in values.items()}


def cards(config):
    e = config.get('episode') or {}
    if not e:
        return []
    text = (f"Whole Demo levels of up to {number(e.get('horizon_seconds'))} s, ending at the exit or when the score runs "
            f"out. {number(e.get('episodes_per_iteration'))} episodes per iteration.")
    if e.get('levels'):
        text += f" Levels: {','.join(str(level) for level in e['levels'])}."
    return [dict(title='Episodes', rows=[], note=text)]
