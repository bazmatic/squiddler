"""Task registry is the only place that knows the available creature/scenario adapters."""
from dataclasses import dataclass, field
import importlib

from learning.inchworm_rewards import SETTINGS as INCHWORM_REWARD_SETTINGS
from learning.flocker_rewards import SETTINGS as FLOCKER_REWARD_SETTINGS
from learning.jelly_rewards import SETTINGS as JELLY_REWARD_SETTINGS
from learning.campaign_rewards import SETTINGS as CAMPAIGN_REWARD_SETTINGS


REPLAY_MODES = ('hooks', 'builtin', None)
SETTINGS_SCOPES = ('listed', 'unclaimed')


@dataclass(frozen=True)
class Capabilities:
    """What the trainer UI may offer for a task; the defaults describe a plain task.

    replay_mode: 'hooks' replays through the task's replay hooks module, and watching an episode first validates it
    with `tools.watch_episode --check`; 'builtin' is the built-in replay path of the tasks tools.ppo_neural trains
    (its episode job list in tools.watch_episode, no hooks module, no pre-watch check); None means the task's runs
    cannot be replayed.
    settings_scope: 'listed' offers the common settings and the task's options; 'unclaimed' also offers every setting
    no other task lists in its options."""
    fresh_weights: bool = True
    actor_damping: bool = False
    replay_mode: str | None = 'hooks'
    settings_scope: str = 'listed'

    def __post_init__(self):
        if self.replay_mode not in REPLAY_MODES:
            raise ValueError(f'unknown replay_mode: {self.replay_mode!r}')
        if self.settings_scope not in SETTINGS_SCOPES:
            raise ValueError(f'unknown settings_scope: {self.settings_scope!r}')


@dataclass(frozen=True)
class Task:
    key: str
    module: str
    label: str
    fresh_defaults: dict = field(default_factory=dict)
    options: tuple = ()
    replay_module: str | None = None
    replay_hooks: str | None = None   # the module learning.replay asks to rebuild, score and draw this task's episodes
    dashboard: str | None = None      # the module declaring this task's charts, see learning.dashboard
    capabilities: Capabilities = field(default_factory=Capabilities)
    start: str | None = None          # the policy a new run starts from (relative to the repo) when no run is named

    def replay(self):
        return importlib.import_module(self.replay_module) if self.replay_module else None

    def adapter(self):
        return importlib.import_module(self.module)


TASKS = {
    'inchworm': Task('inchworm', 'learning.adapters.inchworm_scenario', 'Inchworm',
        dict(horizon=120., workers=4, iterations=5000, episodes=24, bootstrap_episodes=80,
             bootstrap_epochs=4, lr=1e-4, gamma=.997, gae_lambda=.97, budget_seconds=7200, start='', seed=0),
        ('budget_seconds','bootstrap_episodes','bootstrap_epochs','seed') + INCHWORM_REWARD_SETTINGS,
        'learning.replay', 'learning.adapters.inchworm_replay', 'learning.adapters.inchworm_dashboard'),
    'flocker_team': Task('flocker_team', 'learning.adapters.flocker_scenario', 'Flocker team',
        dict(horizon=60., workers=4, iterations=10000, bootstrap_episodes=0),
        ('budget_seconds','seed') + FLOCKER_REWARD_SETTINGS, 'learning.replay',
        'learning.adapters.flocker_replay', 'learning.adapters.flocker_dashboard'),
    'hunter': Task('hunter', 'learning.adapters.hunter', 'Hunter', dashboard='learning.adapters.hunter_dashboard',
        capabilities=Capabilities(fresh_weights=False, actor_damping=False, replay_mode='builtin',
                                  settings_scope='unclaimed')),
    'wallcrawler_team': Task('wallcrawler_team', 'learning.adapters.crawler_scenario', 'Wallcrawler team',
        dict(horizon=60.0,workers=4,iterations=10000),
        ('opponent','budget_seconds','bootstrap_episodes','seed'), 'learning.replay',
        'learning.adapters.crawler_replay', 'learning.adapters.crawler_dashboard',
        Capabilities(fresh_weights=True, actor_damping=True, replay_mode='hooks', settings_scope='listed')),
    # Imitation remains opt-in while teacher warm-start experiments are evaluated.
    'jelly_team': Task('jelly_team', 'learning.adapters.jelly_scenario', 'Jelly team',
        dict(horizon=60.0, workers=4, iterations=10000, bootstrap_episodes=0,
             jelly_boredom_penalty=.02, jelly_no_catch_timeout=25.),
        ('budget_seconds','bootstrap_episodes','bootstrap_epochs','seed') + JELLY_REWARD_SETTINGS, 'learning.replay',
        'learning.adapters.jelly_replay', 'learning.adapters.jelly_dashboard',
        Capabilities(fresh_weights=True, actor_damping=False, replay_mode='hooks', settings_scope='listed')),
    # Trained by the hunter's PPO loop; a new run starts from the shipped hunter pilot, migrated to its senses.
    'campaign_hunter': Task('campaign_hunter', 'learning.adapters.campaign_hunter', 'Campaign Hunter',
        dict(horizon=180.0), CAMPAIGN_REWARD_SETTINGS, dashboard='learning.adapters.campaign_dashboard',
        capabilities=Capabilities(fresh_weights=False, replay_mode='builtin', settings_scope='listed'),
        start='assets/neural/hunter.npz'),
}

# Replayable runs recorded before config.json named its task were all Wallcrawler team runs.
UNTAGGED_REPLAY_TASK = 'wallcrawler_team'
# Runs recorded before config.json named its task were hunter runs.
UNTAGGED_TASK = 'hunter'


def task(key=UNTAGGED_TASK):
    try:
        return TASKS[key]
    except KeyError:
        raise ValueError(f'unknown training task: {key}') from None
