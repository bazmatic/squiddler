"""The Jelly team's starter dashboard: keys from jelly_scenario.summarise(), which metrics() writes into train.log over
all of an iteration's episodes. The per-stage stage<n>_* copies are left to the automatic "Other" section for now."""
from learning.dashboard import Layout, Metric
from learning.jelly_rewards import TABLE, recovered_settings  # noqa: F401 (recovered_settings: see learning.dashboard)


METRICS = (
    Metric('navigation_lesson', 'Navigation lesson', '0: open pursuit; 1: simple barrier; 2: caves. Separate from team/combat stage.',
           section='Navigation', digits=0),
    Metric('idle_fraction', 'Time idle', 'Share of Jelly time in a sustained small area, regardless of throttle; the run records its grace window.',
           section='Navigation', higher_is_better=False),
    Metric('catches_per_minute', 'Catches per minute',
           'Target catches per minute over all episodes. Ship-fight episodes have no target, so they pull it down.',
           unit='/min', section='Outcome', higher_is_better=True, short='Catches / min', digits=1),
    Metric('catch_probability', 'Episodes with a catch',
           'Share of episodes in which the team caught the target at least once.',
           section='Outcome', higher_is_better=True),
    Metric('stuck_fraction', 'Time stuck',
           'Share of Jelly time spent pushing for a whole second while moving less than 10 px.',
           section='Navigation', higher_is_better=False),
    Metric('time_to_contact', 'Time to first contact',
           'Mean seconds to the first catch or ship hit; an episode without one counts its whole length.',
           unit='s', section='Navigation', higher_is_better=False),
    Metric('team_catches_per_minute', 'Team catches per minute',
           'Catches per minute in episodes with two or more Jellies; 0 when there were none.',
           unit='/min', section='Teamwork', higher_is_better=True),
    Metric('angular_spread', 'Flank spread',
           'How evenly the Jellies within 200 px of their quarry surround it: 0 all from one side, 1 an even ring or '
           'pincer. Averaged over the time two or more were that close.',
           section='Teamwork', higher_is_better=True),
    Metric('bunched_fraction', 'Time bunched',
           'Share of Jelly time spent within 40 px of a teammate.',
           section='Teamwork', higher_is_better=False),
    Metric('ship_hits', 'Ship hits',
           'Hull points the Jellies knocked off the training ship, summed over the iteration\'s episodes.',
           section='Ship fight', higher_is_better=True),
    Metric('jelly_deaths', 'Jelly deaths',
           'Jellies killed, summed over the iteration\'s episodes.',
           section='Ship fight', higher_is_better=False),
    Metric('survival_fraction', 'Jelly survival',
           'Share of Jellies still alive at the end of ship-fight episodes; 1 when the iteration had none.',
           section='Ship fight', higher_is_better=True),
)

# Training return first and policy drift last, as for every task, with the Jelly's own sections between them.
LAYOUT = Layout(sections=('Training', 'Outcome', 'Navigation', 'Teamwork', 'Ship fight', 'Policy'),
                strip=('catches_per_minute', 'stuck_fraction'))

# The per-stage copies jelly_scenario.metrics() writes as stage<n>_<key> (its STAGE_KEYS, repeated here so the dashboard
# need not import the game; a test checks they match). Per-stage charts are a later addition.
_STAGE_KEYS = {1: ('episodes', 'catches', 'seconds', 'catches_per_minute', 'stuck_fraction', 'time_to_contact'),
               2: ('episodes', 'catches', 'seconds', 'catches_per_minute', 'stuck_fraction', 'time_to_contact',
                   'angular_spread', 'bunched_fraction'),
               3: ('episodes', 'seconds', 'stuck_fraction', 'time_to_contact', 'ship_hits', 'ship_kills',
                   'jelly_deaths', 'survival_fraction', 'time_to_ship_kill')}
# Raw counters behind the charted rates, and imitation_loss (imitation bootstrap is off for the Jelly).
HIDDEN = ('idle_seconds', 'catches', 'seconds', 'stuck_seconds', 'ship_kills', 'bunched_seconds', 'flank_seconds', 'time_to_ship_kill',
          'mean_episode_return', 'episodes', 'imitation_loss',
          *(f'stage{stage}_{key}' for stage, keys in _STAGE_KEYS.items() for key in keys))

# The Jelly's reward settings, from its reward table (learning/jelly_rewards.py). A run recorded before they were
# settings reports them as not recorded, with these defaults (what it trained with). Stage 3 mixes stage-1 and stage-2
# retention cases into its training, where catches still pay.
REWARD_RULES = TABLE.reward_rules()


def stage(row):
    """The highest stage whose episodes the row's iteration flew (stage<n>_episodes, from jelly_scenario.metrics());
    lower stages are its retention cases."""
    flown = [n for n in _STAGE_KEYS if row.get(f'stage{n}_episodes')]
    return max(flown) if flown else None
