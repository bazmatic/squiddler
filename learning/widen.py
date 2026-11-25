"""A wider recurrent policy that starts out behaving exactly like a narrower one."""
import numpy as np
from learning.policy import RecurrentPolicy


def widen(policy, hidden, embed, seed=0):
    """Carry `policy` into a network with `hidden` memory units and `embed` input units.

    Every old unit keeps its weights and ignores the new ones; the new units read the old ones through small random
    weights, and the output heads give them zero weight. So the wider policy chooses exactly what the narrow one did
    until training finds a use for the extra units."""
    old_hidden, old_embed = policy.hidden, policy.params['embed_w'].shape[0]
    if hidden < old_hidden or embed < old_embed:
        raise ValueError(f'cannot shrink a policy from {old_hidden} memory and {old_embed} input units')
    rng = np.random.default_rng(seed)
    p = policy.params

    def noise(shape, like):
        return (rng.standard_normal(shape)*np.std(like)).astype(np.float32)

    wide = {}
    wide['embed_w'] = np.vstack([p['embed_w'], noise((embed-old_embed, p['embed_w'].shape[1]), p['embed_w'])])
    wide['embed_b'] = np.concatenate([p['embed_b'], np.zeros(embed-old_embed, np.float32)])
    for name, old_cols, cols in (('gru_w_ih', old_embed, embed), ('gru_w_hh', old_hidden, hidden)):
        gates = np.zeros((3*hidden, cols), np.float32)
        for g in range(3):                      # reset, update and candidate blocks
            block = noise((hidden, cols), p[name])
            block[:old_hidden, :old_cols] = p[name][g*old_hidden:(g+1)*old_hidden]
            block[:old_hidden, old_cols:] = 0.   # old units never listen to new ones
            gates[g*hidden:(g+1)*hidden] = block
        wide[name] = gates
    for name in ('gru_b_ih', 'gru_b_hh'):
        bias = np.zeros(3*hidden, np.float32)
        for g in range(3):
            bias[g*hidden:g*hidden+old_hidden] = p[name][g*old_hidden:(g+1)*old_hidden]
        wide[name] = bias
    for head, _ in policy.spec.heads:
        weights = np.zeros((p[head+'_w'].shape[0], hidden), np.float32)
        weights[:, :old_hidden] = p[head+'_w']
        wide[head+'_w'], wide[head+'_b'] = weights, p[head+'_b']
    return RecurrentPolicy(wide, policy.mean, policy.scale, policy.spec)
