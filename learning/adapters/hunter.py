"""Hunter adapter retaining the original recorder, value model and checkpoint contract."""
import numpy as np
from hunter.neural.policy import HEAD_SIZES
from hunter.senses import FEATURE_NAMES, OBSERVATION_VERSION
from learning.policy import PolicySpec
from learning.environment import Snapshot, Transition

SPEC = PolicySpec('hunter', FEATURE_NAMES, tuple(HEAD_SIZES.items()), OBSERVATION_VERSION)

# The rules a hunter run keeps for its whole life, recorded when it starts (config.json 'rules'), whatever its settings
# or the code later say, each with the value a run older than that rule flew with. Random mazes are the exception that
# a resume may take up (tools.ppo_neural.train_rl notes the iteration in the run's history).
RUN_RULES = {'combat_ruleset': 'legacy', 'curved_walls': False, 'weapon_tuning': None, 'random_maze': False}


def add_arguments(parser):
    """The hunter's own options on the trainer's command line (its rewards come from learning.hunter_rewards)."""
    parser.add_argument('--fixed-start', action='store_true',
                        help='start the hunter beside the player on every episode instead of at a random open cell')
    parser.add_argument('--health', type=int, default=0,
                        help='hits the hunter survives before it is destroyed and the episode ends (0: indestructible)')
    parser.add_argument('--decoy-player', action='store_true',
                        help='leave the parked player as a target for enemies (by default they chase only the hunter)')
    parser.add_argument('--after-clear', type=float, default=0.0,
                        help='seconds an episode keeps going after the last enemy dies (less if no crystal is left)')
    parser.add_argument('--panic-score', type=float, default=None,
                        help='panic once the last 5 s score below this (e.g. -10); off when not given')
    parser.add_argument('--panic-temperature', type=float, default=3.0,
                        help='the temperature choices are sampled at while panicking')


def default_levels():
    """None: the tune set (experiments.run.TUNE_LEVELS), which a run then records as no levels chosen."""
    return None


def shaping_from(args):
    """The episode shaping a run's arguments ask for: what fly_rollout is called with."""
    from learning.hunter_rewards import TABLE
    from weapons.catalog import RULESET, CATALOG
    return {**TABLE.from_args(args),
            'horizon': args.horizon,
            'random_start': not args.fixed_start,
            'health': args.health,
            'after_clear': args.after_clear,
            'pursue_hunter': not args.decoy_player,
            'panic_score': args.panic_score,
            'panic_temperature': args.panic_temperature,
            'duel_separation': True, 'random_maze': True,
            'combat_ruleset': RULESET,
            'curved_walls': True,
            'weapon_tuning': CATALOG.snapshot()}


def episode_block(args):
    """A run's episode block, for its dashboard card and its replays."""
    import config
    return {'horizon_seconds': args.horizon, 'after_clear_seconds': args.after_clear,
            'episodes_per_iteration': args.episodes, 'learning_rate': args.lr,
            'health': args.health, 'wall_energy': config.SETTINGS.game.neuralWallEnergy if args.health else 0,
            'levels': args.levels,
            'panic_score': args.panic_score, 'panic_temperature': args.panic_temperature}


def summary(flown):
    """An iteration's numbers for the training log, which learning.adapters.hunter_dashboard charts."""
    from experiments.score import mean_components
    score = mean_components([e['result'] for e in flown])
    arena = [e for e in flown if e.get('rules_arena')]
    blind = sum(e.get('blind_steps', 0) for e in flown)
    unseen = sum(e.get('unseen_steps', 0) for e in flown)
    steps = sum(len(e.get('temperatures', ())) for e in flown)
    panicked = sum(int((np.asarray(e.get('temperatures', ())) > 1.0).sum()) for e in flown)
    return dict(score=score['score'], kills_per_min=score['kills_per_min'],
                wall_quotient_per_min=score['wall_quotient_per_min'],
                hits_taken_per_min=score['hits_taken_per_min'],
                speed_fraction=float(np.mean([e.get('speed_fraction', 0.0) for e in flown])),
                deaths_per_episode=float(np.mean([1.0 if e.get('died') else 0.0 for e in flown])),
                powerups_per_episode=float(np.mean([e.get('powerups', 0) for e in flown])),
                upgraded_fraction=float(np.mean([e.get('upgraded_fraction', 0.0) for e in flown])),
                rules_episodes=len(arena),
                rules_beaten=float(np.mean([1.0 if e.get('rules_beaten') else 0.0 for e in arena])) if arena else 0.0,
                compass_follow=sum(e.get('following_steps', 0) for e in flown) / blind if blind else 0.0,
                panic_fraction=panicked / steps if steps else 0.0,
                hits_per_episode=float(np.mean([e.get('hits', 0) for e in flown])),
                unaimed_shots_per_episode=float(np.mean([e.get('unaimed_shots', 0) for e in flown])),
                bored_fraction=sum(e.get('bored_steps', 0) for e in flown) / unseen if unseen else 0.0)


def environment(**kwargs):
    return HunterEnvironment(**kwargs)


def enabled(x, action, masks):
    return [True]*len(SPEC.heads)


class ExternalPolicy:
    controls = (1,0,0)
    def initial_state(self):
        return None
    def act(self, senses, state, *args, **kwargs):
        return self.controls, state


class HunterEnvironment:
    """Single-agent numeric stepping over the same real frames used by legacy hunter training.

    Legacy full-episode runs preserve the original recorder and rewards without translation.
    The stepping interface uses that recorder too, allowing the common collector to exercise
    this adapter with arbitrary policies. Hunter PPO retains its original actor value head.
    """
    spec = SPEC
    def __init__(self, horizon=90., decision_frames=6, **settings):
        from learning.adapters import hunter
        self.adapter = hunter
        self.horizon, self.decision_frames, self.settings = horizon, decision_frames, settings
        self.frames = None

    def run(self, level, seed, student, **settings):
        from experiments.rollout import fly_rollout
        return fly_rollout(level, seed, student, **settings)

    def reset(self, seed, level=14, **scenario):
        from experiments.harness import episode_frames
        from experiments.rollout import RolloutRecorder
        from experiments.student import StudentSpec
        self.close()
        self.policy = ExternalPolicy()
        self.recorder = RolloutRecorder(self.decision_frames)
        self.frames = episode_frames(level,seed,horizon=self.horizon,
            student=StudentSpec('',decision_frames=self.decision_frames), external_policy=self.policy,
            recorder=self.recorder,pursue_hunter=True,**self.settings,**scenario)
        self.current = next(self.frames)
        self.action_trace = []
        self.result = None
        return self.snapshot()

    def snapshot(self):
        game, ship, flight, _, _ = self.current
        # Preview with a separate sensor: do not update the recorder or the pilot's memory twice.
        from hunter.senses import Senses
        if not hasattr(self,'sensor'):
            self.sensor = Senses(flight.settings)
        x = self.sensor.perceive(ship,flight.surroundings,self.policy.controls,flight.controller.frames/60)
        masks = {k:np.ones(n,bool) for k,n in SPEC.heads}
        return Snapshot({0:x} if ship.active else {},{0:masks} if ship.active else {},x)

    def step(self, actions):
        if self.result is not None:
            raise RuntimeError('step after episode ended')
        before = float(self.recorder.arrays()[2].sum())
        self.policy.controls = tuple(actions[0])
        self.action_trace.append({'0':list(actions[0])})
        for _ in range(self.decision_frames):
            try:
                self.current = next(self.frames)
            except StopIteration as done:
                self.result = done.value
                break
        reward = float(self.recorder.arrays()[2].sum())-before
        terminal = self.result is not None and self.result.end != 'horizon'
        truncated = self.result is not None and self.result.end == 'horizon'
        events = {} if self.result is None else vars(self.result)
        return Transition(self.snapshot(),reward,terminal,truncated,events)

    def close(self):
        from experiments.harness import reset, _clear_overrides
        if self.frames is not None:
            self.frames.close()
            game = self.current[0]
            game._close_hunter();game._close_enemy_hunters();_clear_overrides(game)
            self.frames = None
        self.__dict__.pop('sensor',None)
        reset()
