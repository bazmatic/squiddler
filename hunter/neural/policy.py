"""The student's brain: a small GRU with one softmax head per control, run in plain numpy."""
import numpy as np
from hunter.senses import FEATURE_NAMES, SENSE_SIZE, OBSERVATION_VERSION

HEAD_SIZES = {'turn': 3, 'thrust': 2, 'fire': 2}
_PARAMS = ('embed_w', 'embed_b', 'gru_w_ih', 'gru_w_hh', 'gru_b_ih', 'gru_b_hh',
           'turn_w', 'turn_b', 'thrust_w', 'thrust_b', 'fire_w', 'fire_b')


class InvalidPolicy(Exception):
    """The policy file is missing, damaged, or built for a different senses layout."""


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NeuralPolicy:
    feature_names = FEATURE_NAMES

    def __init__(self, params, mean, scale):
        self.params = {k: np.asarray(params[k], dtype=np.float32) for k in _PARAMS}
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.hidden = self.params['gru_w_hh'].shape[1]

    def initial_state(self):
        from learning.policy import RecurrentPolicy
        return RecurrentPolicy.initial_state(self)

    def step(self, senses, state):
        from learning.policy import RecurrentPolicy
        from learning.policy import PolicySpec
        self.spec = PolicySpec('hunter', self.feature_names, tuple(HEAD_SIZES.items()), OBSERVATION_VERSION)
        return RecurrentPolicy.step(self, senses, state)

    def act(self, senses, state, greedy=True, rng=None, temperature=1.0):
        """Choose the controls. Sampled choices are drawn at `temperature`: above 1 the odds flatten (training's
        panic); the greedy choice is the same at any temperature."""
        logits, new_state = self.step(senses, state)
        picks = []
        for head in ('turn', 'thrust', 'fire'):
            z = (logits[head] - logits[head].max()) / temperature
            probs = np.exp(z) / np.exp(z).sum()
            if greedy:
                picks.append(int(np.argmax(probs)))
            else:
                picks.append(int((rng or np.random.default_rng()).choice(len(probs), p=probs)))
        return (picks[0], bool(picks[1]), bool(picks[2])), new_state

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, feature_names=np.array(self.feature_names),
                 observation_version=OBSERVATION_VERSION,
                 **self.params)

    @classmethod
    def load(cls, path, feature_names=FEATURE_NAMES):
        """Load a policy built for `feature_names`. Only known older layouts of the same pilot are carried over, with
        the inputs added since at zero weight: older, shorter hunter policies for the hunter layout, and the first
        campaign layout for the campaign one."""
        try:
            size = len(feature_names)
            hunter_layout = feature_names == FEATURE_NAMES
            data = np.load(path, allow_pickle=False)
            names = tuple(str(n) for n in data['feature_names'])
            legacy_size = FEATURE_NAMES.index('power_boost') if hunter_layout else None
            version = int(data['observation_version']) if 'observation_version' in data else (1 if len(names) == legacy_size else OBSERVATION_VERSION)
            if version not in ((1, OBSERVATION_VERSION) if hunter_layout else (OBSERVATION_VERSION,)):
                raise InvalidPolicy('unsupported observation version')
            older = older_layouts(feature_names)
            if names != feature_names and names not in older:
                raise InvalidPolicy('senses layout differs from this build')
            params = {k: data[k] for k in _PARAMS}
            mean, scale = data['mean'], data['scale']
            if names != feature_names:
                # Explicit older-layout adapter: the inputs added since have zero influence.
                # It preserves old behaviour but does not teach old pilots to use them.
                width = len(names)
                if params['embed_w'].shape[1] != width or mean.shape != (width,) or scale.shape != (width,):
                    raise InvalidPolicy('invalid legacy observation dimensions')
                extra = size - width
                params['embed_w'] = np.pad(params['embed_w'], ((0,0),(0,extra)))
                mean = np.pad(mean, (0,extra))
                scale = np.pad(scale, (0,extra), constant_values=1)
            policy = cls(params, mean, scale)
            policy.feature_names = feature_names
            policy.legacy_observations = len(names) == legacy_size
            if policy.params['embed_w'].shape[1] != size:
                raise InvalidPolicy('senses layout differs from this build')
            if policy.mean.shape != (size,) or policy.scale.shape != (size,):
                raise InvalidPolicy('standardisation does not fit the senses layout')
            logits, _ = policy.step(np.zeros(size, dtype=np.float32), policy.initial_state())
            if not all(np.isfinite(v).all() and v.shape == (n,) for v, n in
                       ((logits[k], HEAD_SIZES[k]) for k in HEAD_SIZES)):
                raise InvalidPolicy('weights do not run')
            return policy
        except InvalidPolicy:
            raise
        except Exception as exc:
            raise InvalidPolicy(type(exc).__name__) from None


def older_layouts(feature_names):
    """The older layouts a policy file for `feature_names` may be in and still load."""
    from hunter.campaign_layout import CAMPAIGN_FEATURE_NAMES, OLDER_CAMPAIGN_LAYOUTS
    if feature_names == FEATURE_NAMES:
        return (FEATURE_NAMES[:FEATURE_NAMES.index('power_boost')],)
    if feature_names == CAMPAIGN_FEATURE_NAMES:
        return OLDER_CAMPAIGN_LAYOUTS
    return ()


def random_policy(seed, hidden=32, embed=16):
    """A small random policy, for tests."""
    rng = np.random.default_rng(seed)
    g = lambda *shape: (rng.normal(0, 0.3, shape)).astype(np.float32)
    params = {'embed_w': g(embed, SENSE_SIZE), 'embed_b': g(embed),
              'gru_w_ih': g(3 * hidden, embed), 'gru_w_hh': g(3 * hidden, hidden),
              'gru_b_ih': g(3 * hidden), 'gru_b_hh': g(3 * hidden)}
    for head, n in HEAD_SIZES.items():
        params[f'{head}_w'], params[f'{head}_b'] = g(n, hidden), g(n)
    return NeuralPolicy(params, np.zeros(SENSE_SIZE), np.ones(SENSE_SIZE))
